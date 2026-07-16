# 1. 列表操作
fruits = ["apple", "banana", "cherry", "date"]
# - 追加 "elderberry"
fruits.append("elderberry")
# - 把 "banana" 替换成 "blueberry"
fruits.remove("banana")
fruits.insert(1, "blueberry")
# - 删除最后一个元素
fruits.pop()
# - 打印列表
print(f"{fruits}")

# 2. 元组解包
coordinates = (10, 20)
# 解包成 x 和 y，然后打印 f"x={x}, y={y}"
x, y = coordinates
print(f"x = {x}, y = {y}")

# 3. 字典操作
user = {"name": "Bob", "age": 30}
# - 添加 city="Shanghai"
user["city"] = "Shanghai"
# - 修改 age 为 31
user["age"] = 31
# - 用 get 方法取出 age，存在就打印，不存在打印 "Key not found"
result = user.get("age", "Key not found")
print(result)
# - 遍历字典，打印所有键值对
for k, v in user.items():
    print(f"{k}:{v}")

# 4. 集合操作
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
# - 打印并集
print(a | b)
# - 打印交集
print(a & b)
# - 打印 a 和 b 的差集（a 有但 b 没有的）
print(a - b)

# 5. 列表去重（用 set）
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# 用 set 去重，再转回 list，打印结果
print(list(set(numbers)))
