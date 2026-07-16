# 1. 条件判断
score = 78
# 使用 if/elif/else 打印等级 (A: >=90, B: >=80, C: >=70, D: <70)
if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
else:
    print("D")

# 2. for 循环遍历列表
fruits = ["apple", "banana", "cherry"]
# 遍历 fruits，打印 "I like {fruit}"
for fruit in fruits:
    print(f"I like {fruit}")


# 3. 使用 range() 打印 1 到 10 的所有奇数
print(list(i for i in range(1, 10) if i % 2 == 1))


# 4. 列表推导式
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 用列表推导式生成一个新列表，只包含偶数
print(list(o for o in numbers if o % 2 == 0))
# 5. 字典推导式
# 用字典推导式生成 {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print({z: z**2 for z in range(1, 6)})

# 6. break 和 continue
# 遍历 1 到 20，打印所有数字，但遇到 13 时跳过（continue），遇到 18 时提前结束（break）
count = 1
while count <= 20:
    if count == 13:
        count += 1
        continue
    if count == 18:
        break
    print(count)
    count += 1
