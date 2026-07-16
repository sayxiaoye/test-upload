import pytest

from src.core.file_ops import divide, file_exists, read_note, save_note


def test_save_and_read_note(tmp_path):
    # 使用临时目录，不污染项目
    test_file = tmp_path / "test.txt"
    # 保存内容
    save_note(str(test_file), "JUST!Test")
    # 读取内容
    content = read_note(str(test_file))
    # 验证
    assert content == "JUST!Test"


def test_read_note_not_found():
    """测试读取不存在的文件"""
    content = read_note("non_existent_file.txt")
    assert content == "文件不存在"


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


def test_file_exists_true(tmp_path):
    file_path = tmp_path / "exists.txt"
    file_path.write_text("hello", encoding="utf-8")
    assert file_exists(str(file_path)) is True


def test_file_exists_false():
    assert file_exists("non_existent_file.txt") is False


# ============ Fixture 测试 ============
@pytest.fixture
def sample_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is sample content", encoding="utf-8")
    return file_path


def test_read_sample(sample_file):
    content = read_note(str(sample_file))
    assert "sample content" in content
