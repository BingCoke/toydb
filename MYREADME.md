# parse_expression 函数可以解析的位置
- from 后面
- 表达式
- update set c1=(expression(这里可以解析))
- select (expression) from
- form t1 join t2 on (expression)
- 解析clomn 基本都是用expression
- having:BufferLineGoToBuffer 1




# BUG
## group_by sleect * 同时出现不会报错
## 字段名重复没有判断
## inject_hidden 导致的bug 用的就是a的label 但是因为被function包裹 所以出问题了
```
toydb> select max(id) v ,  studio_id a, sum(studio_id) from movies group by studio_id ;
v|a|?
2|2|2
12|5|15
10|4|20
6|1|2
3|3|3
toydb> select max(id) v ,  studio_id a, sum(studio_id) from movies group by studio_id having sum(a)>0;
v|a|?
```
1. orderby having 需要分开进行 orderby 不能聚合函数
2. having 需要知道label 是不是group by 字段,expr一样就不管了 select的时候肯定会排查的 
所以label的时候需要再把expression转换回来 让select的时候去判断

改完可以了
```
toydb> select max(id) v ,  studio_id a, sum(studio_id) from movies group by studio_id having sum(a)>2;
v|a|?
12|5|15
10|4|20
3|3|3
toydb> select max(id) v ,  studio_id a, sum(studio_id) from movies group by studio_id having sum(a)>0;
v|a|?
3|3|3
2|2|2
10|4|20
6|1|2
12|5|15
toydb>
```


# feature
- 单主键索引
