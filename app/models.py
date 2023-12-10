from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import date


class RevenueBase(SQLModel):
    date: date
    revenue: int


class Revenue(RevenueBase, table=True):
    id: Optional[int] = Field(default=None, nullable=False, primary_key=True)


class RevenueCreate(RevenueBase):
    pass


class RevenueRead(RevenueBase):
    id: int


class RevenueUpdate(SQLModel):
    date: Optional[date]
    revenue: Optional[int]
