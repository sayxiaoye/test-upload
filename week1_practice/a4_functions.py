# 1. 写一个函数 greet_user(name, greeting="Hello")，返回 f"{greeting}, {name}!"
# 测试调用：不传 greeting 参数、传一个自定义 greeting
def greet_user(name, greeting="Hello"):
    return f"{greeting}, {name}"


print(greet_user("yes", "wa"))


# 2. 写一个函数 calculate_total(price, tax_rate=0.05)，返回 price + price * tax_rate
# 测试：price=100 不传 tax_rate，price=200 传 tax_rate=0.08
def calculate_total(price: float, tax_rate: float = 0.05) -> float:
    return price + price * tax_rate


print(calculate_total(100))
print(calculate_total(200, 0.08))


# 3. 写一个函数 sum_all(*numbers)，返回所有数字的和
# 测试：sum_all(1, 2, 3, 4)
def sum_all(*numbers):
    print(sum(numbers))


sum_all(*range(1, 5))

# 4. 写一个函数 print_user(**info)，打印所有键值对
# 测试：print_user(name="Tom", age=25, city="Shanghai")


def print_user(**info):
    for v, k in info.items():
        print(f"{v}:{k}")


print_user(name="Tom", age=25, city="Shanghai")


# 5. 写一个函数 calculate_average(numbers)，返回列表的平均值
# 提示：sum(numbers) / len(numbers)
# 测试：calculate_average([10, 20, 30, 40])
def calculate_average(numbers):
    print(sum(numbers) / len(numbers))


calculate_average([10, 20, 30, 40])
