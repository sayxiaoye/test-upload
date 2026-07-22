"""
数据库模块
使用SQLAlchemy ORM管理数据
"""

from typing import cast

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


# 创建基类
class Base(DeclarativeBase):
    """ORM模型基类"""

    pass


# ============ 模型定义 ============
class Document(Base):
    """文档表"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(25), nullable=False)
    filename = Column(String(255), nullable=False, unique=True)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 关系： 一个文档有多条查询记录
    queries = relationship("QueryLog", back_populates="document")

    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title})>"


class QueryLog(Base):
    """查询日志表"""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(String(500), nullable=False)
    response_text = Column(String(2000), nullable=True)
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    score = Column(Float, nullable=True)
    create_at = Column(DateTime, default=func.now())

    # 关系
    document = relationship("Document", back_populates="queries")

    def __repr__(self):
        return f"<QueryLog(id={self.id}, query={self.query_text[:30]}...)>"


# ============ 数据库操作类 ============
class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: str = "data/app.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._create_tables()

    def _create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()

    # ----- 文档操作 -----
    def create_document(
        self, title: str, filename: str, file_type: str, file_size: int | None = None
    ) -> Document:
        """创建文档记录"""
        with self.get_session() as session:
            doc = Document(
                title=title,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    def get_document(self, doc_id: int) -> Document | None:
        """根据ID 获取文档"""
        with self.get_session() as session:
            result = session.query(Document).filter(Document.id == doc_id).first()
            return cast(Document | None, result)

    def get_document_by_filename(self, filename: str) -> Document | None:
        """根据name获取文档"""
        with self.get_session() as session:
            result = (
                session.query(Document).filter(Document.filename == filename).first()
            )
            return cast(Document | None, result)

    def list_documents(self, limit: int = 10) -> list[Document]:
        """列出所有文档"""
        with self.get_session() as session:
            result = (
                session.query(Document)
                .order_by(Document.created_at.desc())
                .limit(limit)
                .all()
            )
            return cast(list[Document], result)

    def update_document(self, doc_id: int, **kwargs) -> Document | None:
        """update文档"""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                for key, value in kwargs.items():
                    if hasattr(doc, key):
                        setattr(doc, key, value)
                session.commit()
                session.refresh(doc)
            return cast(Document | None, doc)

    def delete_document(self, doc_id: int) -> bool:
        """delete文档"""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False

    # ----- 查询日志操作 -----
    def create_query_log(
        self,
        query_text: str,
        response_text: str | None = None,
        source_document_id: int | None = None,
        score: float | None = None,
    ) -> QueryLog:
        """创建查询日志"""
        with self.get_session() as session:
            log = QueryLog(
                query_text=query_text,
                response_text=response_text,
                source_document_id=source_document_id,
                score=score,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log

    def list_query_logs(self, limit: int = 20) -> list[QueryLog]:
        """列出查询日志"""
        with self.get_session() as session:
            result = (
                session.query(QueryLog)
                .order_by(QueryLog.create_at.desc())
                .limit(limit)
                .all()
            )
            return cast(list[QueryLog], result)

    def get_query_stats(self) -> dict:
        """获取调查统计"""
        with self.get_session() as session:
            total = session.query(QueryLog).count()
            avg_score = session.query(func.avg(QueryLog.score)).scalar() or 0
            return {"total_queries": total, "avg_score": round(avg_score, 2)}


if __name__ == "__main__":
    import os

    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    db = DatabaseManager()

    print("=" * 60)
    print("📊 数据库操作演示")
    print("=" * 60)

    # 1. 创建文档
    print("\n📄 创建文档:")
    doc1 = db.create_document(
        title="Python 学习笔记",
        filename="python_notes.pdf",
        file_type="pdf",
        file_size=1024000,
    )
    print(f"  ID: {doc1.id}, 标题: {doc1.title}")

    doc2 = db.create_document(
        title="RAG 入门指南",
        filename="rag_guide.docx",
        file_type="docx",
        file_size=512000,
    )
    print(f"  ID: {doc2.id}, 标题: {doc2.title}")

    # 2. 查询文档
    print("\n🔍 查询文档:")
    doc = db.get_document(1)
    if doc:
        print(f"  文档 1: {doc.title}")
    else:
        print("  文档 1: 未找到")

    # 3. 列出所有文档
    print("\n📋 所有文档:")
    for d in db.list_documents():
        print(f"  - {d.id}: {d.title} ({d.file_type})")

    # 4. 创建查询日志
    print("\n📝 创建查询日志:")
    log1 = db.create_query_log(
        query_text="什么是 RAG？",
        response_text="RAG 是检索增强生成...",
        source_document_id=2,
        score=4.5,
    )
    print(f"  查询日志 ID: {log1.id}")

    log2 = db.create_query_log(
        query_text="Python 虚拟环境怎么用？",
        response_text="Python 虚拟环境使用 venv...",
        source_document_id=1,
        score=5.0,
    )
    print(f"  查询日志 ID: {log2.id}")

    # 5. 列出查询日志
    print("\n📋 查询日志:")
    for log in db.list_query_logs():
        print(f"  - {log.query_text[:30]}... (评分: {log.score})")

    # 6. 统计信息
    print("\n📊 统计信息:")
    stats = db.get_query_stats()
    print(f"  总查询数: {stats['total_queries']}")
    print(f"  平均评分: {stats['avg_score']}")

    print("\n✅ 数据库操作演示完成!")
