use super::super::schema::Catalog;
use super::super::types::{Expression, Value};
use super::Node;
use crate::error::Result;

use std::mem::replace;

/// A plan optimizer
pub trait Optimizer {
    fn optimize(&self, node: Node) -> Result<Node>;
}

/// A constant folding optimizer, which replaces constant expressions with their evaluated value, to
/// prevent it from being re-evaluated over and over again during plan execution.
/// 计算常量 比如 3+2 这里就会被优化成为 5
pub struct ConstantFolder;

impl Optimizer for ConstantFolder {
    fn optimize(&self, node: Node) -> Result<Node> {
        node.transform(&|n| Ok(n), &|n| {
            n.transform_expressions(
                &|e| {
                    // 如果树中有field,那么就不转换了，有就转换
                    // 感觉可以转，但是需要重写evaluate函数
                    if !e.contains(&|expr| matches!(expr, Expression::Field(_, _))) {
                        Ok(Expression::Constant(e.evaluate(None)?))
                    } else {
                        Ok(e)
                    }
                },
                &|e| Ok(e),
            )
        })
    }
}

/// A filter pushdown optimizer, which moves filter predicates into or closer to the source node.
/// 过滤下推优化器 将过滤断言 放进或者更靠近sourse node
pub struct FilterPushdown;

impl Optimizer for FilterPushdown {
    fn optimize(&self, node: Node) -> Result<Node> {
        node.transform(
            &|n| match n {
                Node::Filter { mut source, predicate } => {
                    // We don't replace the filter node here, since doing so would cause transform()
                    // to skip the source as it won't reapply the transform to the "same" node.
                    // We leave a noop filter node instead, which will be cleaned up by NoopCleaner.
                    if let Some(remainder) = self.pushdown(predicate, &mut *source) {
                        Ok(Node::Filter { source, predicate: remainder })
                    } else {
                        Ok(Node::Filter {
                            source,
                            predicate: Expression::Constant(Value::Boolean(true)),
                        })
                    }
                }
                Node::NestedLoopJoin {
                    mut left,
                    left_size,
                    mut right,
                    predicate: Some(predicate),
                    outer,
                } => {
                    // 把左右表的等值连接 推向source node
                    let predicate = self.pushdown_join(predicate, &mut left, &mut right, left_size);
                    Ok(Node::NestedLoopJoin { left, left_size, right, predicate, outer })
                }
                n => Ok(n),
            },
            &|n| Ok(n),
        )
    }
}

impl FilterPushdown {
    /// Attempts to push an expression down into a target node, returns any remaining expression.
    fn pushdown(&self, mut expression: Expression, target: &mut Node) -> Option<Expression> {
        match target {
            Node::Scan { ref mut filter, .. } => {
                if let Some(filter) = filter.take() {
                    expression = Expression::And(Box::new(expression), Box::new(filter))
                }
                filter.replace(expression)
            }
            Node::NestedLoopJoin { ref mut predicate, .. } => {
                if let Some(predicate) = predicate.take() {
                    expression = Expression::And(Box::new(expression), Box::new(predicate));
                }
                predicate.replace(expression)
            }
            Node::Filter { ref mut predicate, .. } => {
                let p = replace(predicate, Expression::Constant(Value::Null));
                *predicate = Expression::And(Box::new(p), Box::new(expression));
                None
            }
            _ => Some(expression),
        }
    }

    /// Attempts to partition a join predicate and push parts of it down into either source,
    /// returning any remaining expression.
    /// 假设我们有两个表 A 和 B，其中 A 有两列 id 和 name，B 有两列 id 和 age，现在我们要将这两个表按照 id 进行连接，即 A.id = B.id。
    /// 现在我们有一个查询语句：SELECT * FROM A JOIN B ON A.id = B.id AND A.name = 'John'。
    /// 首先，我们需要将 A 和 B 的条件都转换成 CNF。这里的条件是 A.id = B.id AND A.name = 'John'，我们可以将它转化为 (A.id = B.id) AND (A.name = 'John')。
    /// 接下来，我们需要将 CNF 中的条件分为只涉及 A 表和只涉及 B 表的两组条件。
    /// 对于上面的条件，第一项涉及到了 A 和 B 表，因此它不能被分组，但是第二项只涉及 A 表，因此它可以被分到 push_left 中。
    /// 现在我们需要找出 A.id 和 B.id 两列之间是否存在常量关系，即是否存在 A.id = 1 或者 B.id = 1 这样的条件。
    /// 假设存在 A.id = 1，那么我们就可以将 1 这个常量值转移到 B.id 这一侧，变成 B.id = 1，这样就可以在 B 表上进行索引优化。
    /// 为了实现这个转移过程，我们需要遍历 CNF 中的所有条件，找出形如 A.id = B.id 的等值条件，并检查 A.id 和 B.id 是否与某个常量关联。
    /// 如果 A.id 关联了常量，那么我们将常量值加入 push_right 中，构造一个新的表达式 B.id = 1，并添加到 CNF 中；
    /// 如果 B.id 关联了常量，那么我们将常量值加入 push_left 中，构造一个新的表达式 A.id = 1，并添加到 CNF 中。
    /// 这样做的目的是为了在两个表中都进行索引优化，提高查询的效率。
    /// 这里只是一个例子，右表其实是右节点，会继续下沉。
    fn pushdown_join(
        &self,
        predicate: Expression,
        left: &mut Node,
        right: &mut Node,
        boundary: usize,
    ) -> Option<Expression> {
        // Convert the predicate into conjunctive normal form, and partition into expressions
        // only referencing the left or right sources, leaving cross-source expressions.
        // 全部变成 and 连接每个表达式
        let cnf = predicate.into_cnf_vec();

        let (mut push_left, cnf): (Vec<Expression>, Vec<Expression>) =
            cnf.into_iter().partition(|e| {
                // Partition only if no expressions reference the right-hand source.
                // 将包含右表的全部清除，push_left 最终全是只包含 left表字段的表达式
                !e.contains(&|e| matches!(e, Expression::Field(i, _) if i >= &boundary))
            });
        let (mut push_right, mut cnf): (Vec<Expression>, Vec<Expression>) =
            cnf.into_iter().partition(|e| {
                // Partition only if no expressions reference the left-hand source.
                // 将包含左表字段的清楚，push_right最终只包含 right字段表达式
                !e.contains(&|e| matches!(e, Expression::Field(i, _) if i < &boundary))
            });

        // Look for equijoins that have constant lookups on either side, and transfer the constants
        // to the other side of the join as well. This allows index lookup optimization in both
        // sides. We already know that the remaining cnf expressions span both sources.
        // 此时cnf中表达式是同时包含左右字段的
        for e in &cnf {
            if let Expression::Equal(ref lhs, ref rhs) = e {
                if let (Expression::Field(l, ln), Expression::Field(r, rn)) = (&**lhs, &**rhs) {
                    // 刚刚已经分分流过了，所以有两个filed的一定是左右两表，只要把左右分出来就行
                    let (l, ln, r, rn) = if l > r { (r, rn, l, ln) } else { (l, ln, r, rn) };

                    // 先将 左边等值查找的值找出来 然后去查 push_right 中是否有 filedl=常量A
                    // 记住 filedl = filedr 如果找到了，就可以尝试把filedr=常量A 放进 push_right
                    // 这样在右边执行的时候也可以进行索引优化
                    if let Some(lvals) = push_left.iter().find_map(|e| e.as_lookup(*l)) {
                        push_right.push(Expression::from_lookup(*r, rn.clone(), lvals));
                    } else if let Some(rvals) = push_right.iter().find_map(|e| e.as_lookup(*r)) {
                        push_left.push(Expression::from_lookup(*l, ln.clone(), rvals));
                    }
                }
            }
        }

        // Push predicates down into the sources.
        // 把左表中等值查询下沉到source node
        if let Some(push_left) = Expression::from_cnf_vec(push_left) {
            if let Some(remainder) = self.pushdown(push_left, left) {
                cnf.push(remainder)
            }
        }

        if let Some(mut push_right) = Expression::from_cnf_vec(push_right) {
            // All field references to the right must be shifted left.
            push_right = push_right
                .transform(
                    &|e| match e {
                        // 做下推之前要把下标排一下，不需要左表下标了，
                        Expression::Field(i, label) => Ok(Expression::Field(i - boundary, label)),
                        e => Ok(e),
                    },
                    &|e| Ok(e),
                )
                .unwrap();
            if let Some(remainder) = self.pushdown(push_right, right) {
                cnf.push(remainder)
            }
        }

        Expression::from_cnf_vec(cnf)
    }
}

/// An index lookup optimizer, which converts table scans to index lookups.
/// 把全表扫描替换成索引
pub struct IndexLookup<'a, C: Catalog> {
    catalog: &'a mut C,
}

impl<'a, C: Catalog> IndexLookup<'a, C> {
    pub fn new(catalog: &'a mut C) -> Self {
        Self { catalog }
    }

    // Wraps a node in a filter for the given CNF vector, if any, otherwise returns the bare node.
    fn wrap_cnf(&self, node: Node, cnf: Vec<Expression>) -> Node {
        if let Some(predicate) = Expression::from_cnf_vec(cnf) {
            Node::Filter { source: Box::new(node), predicate }
        } else {
            node
        }
    }
}

impl<'a, C: Catalog> Optimizer for IndexLookup<'a, C> {
    fn optimize(&self, node: Node) -> Result<Node> {
        node.transform(&|n| Ok(n), &|n| match n {
            Node::Scan { table, alias, filter: Some(filter) } => {
                let columns = self.catalog.must_read_table(&table)?.columns;
                let pk = columns.iter().position(|c| c.primary_key).unwrap();

                // Convert the filter into conjunctive normal form, and try to convert each
                // sub-expression into a lookup. If a lookup is found, return a lookup node and then
                // apply the remaining conjunctions as a filter node, if any.
                let mut cnf = filter.clone().into_cnf_vec();
                for i in 0..cnf.len() {
                    // 如果是主键索引 并且等值 那么先将索引都提取出来，如何再包一层 filter
                    if let Some(keys) = cnf[i].as_lookup(pk) {
                        cnf.remove(i);
                        return Ok(
                            self.wrap_cnf(Node::KeyLookup { table, alias, keys }, cnf)
                            );
                    }
                    for (ci, column) in columns.iter().enumerate().filter(|(_, c)| c.index) {
                        if let Some(values) = cnf[i].as_lookup(ci) {
                            cnf.remove(i);
                            // 如果是索引就包一层索引
                            return Ok(self.wrap_cnf(
                                Node::IndexLookup {
                                    table,
                                    alias,
                                    column: column.name.clone(),
                                    values,
                                },
                                cnf,
                            ));
                        }
                    }
                }
                // 如果还是没有什么变化就算了
                Ok(Node::Scan { table, alias, filter: Some(filter) })
            }
            n => Ok(n),
        })
    }
}

/// Cleans up noops, e.g. filters with constant true/false predicates.
/// FIXME This should perhaps replace nodes that can never return anything with a Nothing node,
/// but that requires propagating the column names.
pub struct NoopCleaner;

impl Optimizer for NoopCleaner {
    fn optimize(&self, node: Node) -> Result<Node> {
        use Expression::*;
        node.transform(
            // While descending the node tree, clean up boolean expressions.
            &|n| {
                n.transform_expressions(&|e| Ok(e), &|e| match &e {
                    And(lhs, rhs) => match (&**lhs, &**rhs) {
                        (Constant(Value::Boolean(false)), _)
                        | (Constant(Value::Null), _)
                        | (_, Constant(Value::Boolean(false)))
                        | (_, Constant(Value::Null)) => Ok(Constant(Value::Boolean(false))),
                        (Constant(Value::Boolean(true)), e)
                        | (e, Constant(Value::Boolean(true))) => Ok(e.clone()),
                        _ => Ok(e),
                    },
                    Or(lhs, rhs) => match (&**lhs, &**rhs) {
                        (Constant(Value::Boolean(false)), e)
                        | (Constant(Value::Null), e)
                        | (e, Constant(Value::Boolean(false)))
                        | (e, Constant(Value::Null)) => Ok(e.clone()),
                        (Constant(Value::Boolean(true)), _)
                        | (_, Constant(Value::Boolean(true))) => Ok(Constant(Value::Boolean(true))),
                        _ => Ok(e),
                    },
                    // No need to handle Not, constant folder should have evaluated it already.
                    _ => Ok(e),
                })
            },
            // While ascending the node tree, remove any unnecessary filters or nodes.
            // FIXME This should replace scan and join predicates with None as well.
            &|n| match n {
                Node::Filter { source, predicate } => match predicate {
                    Expression::Constant(Value::Boolean(true)) => Ok(*source),
                    predicate => Ok(Node::Filter { source, predicate }),
                },
                n => Ok(n),
            },
        )
    }
}

// Optimizes join types, currently by swapping nested-loop joins with hash joins where appropriate.
pub struct JoinType;

// 如果是两个字段等于，的join连接，那么就拆开成为hash连接
impl Optimizer for JoinType {
    fn optimize(&self, node: Node) -> Result<Node> {
        node.transform(
            &|n| match n {
                // Replace nested-loop equijoins with hash joins.
                Node::NestedLoopJoin {
                    left,
                    left_size,
                    right,
                    predicate: Some(Expression::Equal(a, b)),
                    outer,
                } => match (*a, *b) {
                    (Expression::Field(a, a_label), Expression::Field(b, b_label)) => {
                        let (left_field, right_field) = if a < left_size {
                            ((a, a_label), (b - left_size, b_label))
                        } else {
                            ((b, b_label), (a - left_size, a_label))
                        };
                        Ok(Node::HashJoin { left, left_field, right, right_field, outer })
                    }
                    (a, b) => Ok(Node::NestedLoopJoin {
                        left,
                        left_size,
                        right,
                        predicate: Some(Expression::Equal(a.into(), b.into())),
                        outer,
                    }),
                },
                n => Ok(n),
            },
            &|n| Ok(n),
        )
    }
}
