# 1. 定义变量
name = "Python Learner"
age = 25
score = 88.594949429239339
is_pass = True

# 2. 打印所有变量的类型
# 提示：使用 type() 函数

print(type(name))
print(type(age))
print(type(score))
print(type(is_pass))


# 3. 计算并打印：年龄增加 1 岁后的值

oneYear = age + 1

print(f"年龄增加1岁后 {oneYear}")

# 4. 将 score 转换为整数（去掉小数），打印结果
score = 88.594949429239339
print(f"{score // 1:.5f}")
print(int(score))
print(f"{score:.4f}")
print(f"{score:.2f}")

# 5. 定义一个变量 result = None，判断它是否为 None
# 提示：用 is 运算符：result is None

result = None

print(result is None)
