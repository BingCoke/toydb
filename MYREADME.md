# parse_expression 函数可以解析的位置
- from 后面
- 表达式
- update set c1=(expression(这里可以解析))
- select (expression) from
- form t1 join t2 on (expression)
- 解析clomn 基本都是用expression
- having:BufferLineGoToBuffer 1



# BUG
- group_by sleect * 同时出现不会报错
- 字段名重复没有判断
# feature
- 但主键索引
