from sqlalchemy import Column, Integer, Text, String, Float, LargeBinary, CheckConstraint, ForeignKey, DateTime, func
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    
    id = Column(Integer, primary_key=True)
    username = Column(Text, unique=True, nullable=False, index=True)
    birth_year = Column(Integer)
    birth_month = Column(Integer)
    gender = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint('birth_year BETWEEN 1900 AND EXTRACT(YEAR FROM CURRENT_DATE)', name='check_birth_year'),
        CheckConstraint('birth_month BETWEEN 1 AND 12', name='check_birth_month'),
        CheckConstraint("gender IN ('male', 'female', 'other', 'prefer_not_to_say')", name='check_gender'),
    )


class Report(Base):
    __tablename__ = "report"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    report_type = Column(String, nullable=False)  # 'text_only' or 'medical_image'
    raw_text = Column(Text)
    overall_interpretation = Column(Text, nullable=True)  # LLM generated interpretation for the entire report
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("report_type IN ('text', 'medical_image')", name='check_report_type'),
    )


class LabResult(Base):
    __tablename__ = "lab_result"

    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("report.id", ondelete="CASCADE"), nullable=False, index=True)
    test_name = Column(String, nullable=False)
    consolidated_test_name = Column(String, nullable=False, index=True)
    result_value = Column(String)
    doc_comment = Column(Text, nullable=True)
    unit = Column(String, nullable=True)  # e.g., 'mg/dL', 'mmol/L'
    lower_range = Column(Float, nullable=True)  # llm suggested
    upper_range = Column(Float, nullable=True)  # llm suggested
    interpretation = Column(String)  # 'normal', 'high', 'low'
    test_date = Column(DateTime)
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Added for consistency


class MedicalImage(Base):
    __tablename__ = "medical_image"

    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("report.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)
    image_type = Column(String, nullable=False)  # e.g., 'X-ray', 'MRI', 'CT', 'Graph'
    image_descriptions = Column(Text)
    image_data = Column(LargeBinary)  # store a file path or encoded string
    image_interpretation = Column(Text, nullable=True)  # LLM generated interpretation for the image
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Embeddings(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("report.id", ondelete="CASCADE"), nullable=False, index=True)
    source_table = Column(String, nullable=False)  # 'report', 'lab_results', 'images'
    source_row_id = Column(Integer, nullable=False)
    content = Column(Text)
    embedding = Column(Vector(1536))  # Change dim if using a different model
    chunk_index = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("source_table IN ('report', 'lab_result', 'medical_image')", name='check_source_table'),
    )


# class HealthStateSnapshot(Base):