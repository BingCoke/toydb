use super::super::engine::Transaction;
use super::super::types::{Expression, Rows};
use super::{Executor, ResultSet, Row, Value};
use crate::error::{Error, Result};

use std::collections::HashMap;

/// A nested loop join executor, which checks each row in the left source against every row in
/// the right source using the given predicate.
pub struct NestedLoopJoin<T: Transaction> {
    left: Box<dyn Executor<T>>,
    right: Box<dyn Executor<T>>,
    predicate: Option<Expression>,
    outer: bool,
}

impl<T: Transaction> NestedLoopJoin<T> {
    pub fn new(
        left: Box<dyn Executor<T>>,
        right: Box<dyn Executor<T>>,
        predicate: Option<Expression>,
        outer: bool,
    ) -> Box<Self> {
        Box::new(Self { left, right, predicate, outer })
    }
}

impl<T: Transaction> Executor<T> for NestedLoopJoin<T> {
    fn execute(self: Box<Self>, txn: &mut T) -> Result<ResultSet> {
        if let ResultSet::Query { mut columns, rows } = self.left.execute(txn)? {
            if let ResultSet::Query { columns: rcolumns, rows: rrows } = self.right.execute(txn)? {
                let right_width = rcolumns.len();
                columns.extend(rcolumns);
                // FIXME Since making the iterators or sources clonable is non-trivial (requiring
                // either avoiding Rust standard iterators or making sources generic), we simply
                // fetch the entire right result as a vector.
                // 效率极低的代码段
                return Ok(ResultSet::Query {
                    rows: Box::new(NestedLoopRows::new(
                        rows,
                        rrows.collect::<Result<Vec<_>>>()?,
                        right_width,
                        self.predicate,
                        self.outer,
                    )),
                    columns,
                });
            }
        }
        Err(Error::Internal("Unexpected result set".into()))
    }
}

struct NestedLoopRows {
    left: Rows,
    left_row: Option<Result<Row>>,
    right: Box<dyn Iterator<Item = Row> + Send>,
    right_vec: Vec<Row>,
    right_empty: Vec<Value>,
    right_hit: bool,
    predicate: Option<Expression>,
    outer: bool,
}

impl NestedLoopRows {
    fn new(
        mut left: Rows,
        right: Vec<Row>,
        right_width: usize,
        predicate: Option<Expression>,
        outer: bool,
    ) -> Self {
        Self {
            left_row: left.next(),
            left,
            // right迭代器
            right: Box::new(right.clone().into_iter()),
            // right的数组
            right_vec: right,
            right_empty: std::iter::repeat(Value::Null).take(right_width).collect(),
            right_hit: false,
            predicate,
            outer,
        }
    }

    // Tries to get the next joined row, with error handling.
    // 大致就是说 先找到left_row然后通过predicate判断右行，找到了就返回
    // 如果找不到说明right没有匹配的了，就left.next(),此时right迭代器以及迭代完了
    // 通过right_vec 复制一个新的迭代器
    // right_hit 表示当前这个left是否有匹配项，每次新的left_row都变成false
    // 如果这个left_row有找到一个right匹配项就会置为true
    // 所以到最后left_row都没有找到匹配项，right_hit就会一直是false
    fn try_next(&mut self) -> Result<Option<Row>> {
        // While there is a valid left row, look for a right-hand match to return.
        while let Some(Ok(left_row)) = self.left_row.clone() {
            // If there is a hit in the remaining right rows, return it.
            if let Some(row) = self.try_next_hit(&left_row)? {
                self.right_hit = true;
                return Ok(Some(row));
            }

            // Otherwise, continue with the next left row and reset the right source.
            // 如果找不到就left下一个
            self.left_row = self.left.next();
            self.right = Box::new(self.right_vec.clone().into_iter());

            // If this is an outer join, when we reach the end of the right items without a hit,
            // we should return a row with nulls for the right fields.
            // 实在找不到匹配项之后， 如果是外连接就补一段空的数组
            if self.outer && !self.right_hit {
                let mut row = left_row;
                row.extend(self.right_empty.clone());
                return Ok(Some(row));
            }
            self.right_hit = false;
        }
        self.left_row.clone().transpose()
    }

    /// Tries to find the next combined row that matches the predicate in the remaining right rows.
    fn try_next_hit(&mut self, left_row: &[Value]) -> Result<Option<Row>> {
        for right_row in &mut self.right {
            let mut row = left_row.to_vec();
            row.extend(right_row);
            // 左右匹配 尝试找到一个匹配项
            if let Some(predicate) = &self.predicate {
                match predicate.evaluate(Some(&row))? {
                    Value::Boolean(true) => return Ok(Some(row)),
                    Value::Boolean(false) => {}
                    Value::Null => {}
                    value => {
                        return Err(Error::Value(format!(
                            "Join predicate returned {}, expected boolean",
                            value
                        )))
                    }
                }
            } else {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
}

impl Iterator for NestedLoopRows {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        self.try_next().transpose()
    }
}

/// A hash join executor
pub struct HashJoin<T: Transaction> {
    left: Box<dyn Executor<T>>,
    left_field: usize,
    right: Box<dyn Executor<T>>,
    right_field: usize,
    outer: bool,
}

impl<T: Transaction> HashJoin<T> {
    pub fn new(
        left: Box<dyn Executor<T>>,
        left_field: usize,
        right: Box<dyn Executor<T>>,
        right_field: usize,
        outer: bool,
    ) -> Box<Self> {
        Box::new(Self { left, left_field, right, right_field, outer })
    }
}
/// HashJoin 条件有两个字段相等的时候进行 m.id=n.id
impl<T: Transaction> Executor<T> for HashJoin<T> {
    fn execute(self: Box<Self>, txn: &mut T) -> Result<ResultSet> {
        if let ResultSet::Query { mut columns, rows } = self.left.execute(txn)? {
            if let ResultSet::Query { columns: rcolumns, rows: rrows } = self.right.execute(txn)? {
                let (l, r, outer) = (self.left_field, self.right_field, self.outer);
                // 把右表中 需要比较的字段作为key,整行作为vaule
                let right: HashMap<Value, Row> = rrows
                    .map(|res| match res {
                        Ok(row) if row.len() <= r => {
                            Err(Error::Internal(format!("Right index {} out of bounds", r)))
                        }
                        Ok(row) => Ok((row[r].clone(), row)),
                        Err(err) => Err(err),
                    })
                    .collect::<Result<_>>()?;
                let empty = std::iter::repeat(Value::Null).take(rcolumns.len());
                columns.extend(rcolumns);
                
                let rows = Box::new(rows.filter_map(move |res| match res {
                    Ok(row) if row.len() <= l => {
                        Some(Err(Error::Value(format!("Left index {} out of bounds", l))))
                    }
                    // 左边需要比较的字段值去找map中是否有相等的，如果有就返回
                    // 没有就看看是不是外连接，外连接需要进行补空
                    Ok(mut row) => match right.get(&row[l]) {
                        Some(hit) => {
                            row.extend(hit.clone());
                            Some(Ok(row))
                        }
                        None if outer => {
                            row.extend(empty.clone());
                            Some(Ok(row))
                        }
                        None => None,
                    },
                    Err(err) => Some(Err(err)),
                }));
                return Ok(ResultSet::Query { columns, rows });
            }
        }
        Err(Error::Internal("Unexpected result set".into()))
    }
}
