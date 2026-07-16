# 1. 模块导入
# 导入 math 模块，用 sqrt 计算 144 的平方根
from math import sqrt

print(sqrt(144))
# 从 datetime 导入 datetime，打印当前时间
from datetime import datetime

print(datetime.now())


# 2. 创建类 Book
# - __init__: title, author, price
# - get_info(): 返回 "《title》by author，¥price"
# - 魔法方法 __str__
class Book:
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def get_info(self):
        return f"《{self.title}》by {self.author}, ${self.price}"

    def __str__(self):
        return self.get_info()


# 3. 创建子类 EBook(Book)
# - 新增属性 format（'PDF' 或 'EPUB'）
# - 重写 get_info()，添加格式信息
class EBook(Book):

    def __init__(self, title, author, price, format):
        super().__init__(title, author, price)
        self.format = format

    def get_info(self):
        return f"《{self.title}》by {self.author}, ${self.price}, {self.format}"


# 4. __name__ == "__main__" 测试
# 把测试代码放在 if __name__ == "__main__": 下

if __name__ == "__main__":
    # test Book
    b = Book("好东西", "andy", "30")
    print(b.get_info())
    eb = EBook("huai", "xixi", "22", "PDF")
    print(eb.get_info())
