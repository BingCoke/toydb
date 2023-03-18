# parse_expression 函数可以解析的位置
- from 后面
- 表达式
- update set c1=(expression(这里可以解析))
- select (expression) from
- form t1 join t2 on (expression)
- 解析clomn 基本都是用expression
- having:BufferLineGoToBuffer 1



# BUG
- planner.rs 695行数据似乎有bug 有前缀限定列名会出现问题
