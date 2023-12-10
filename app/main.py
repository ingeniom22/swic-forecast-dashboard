import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from fastapi.params import Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import plotly
import plotly.express as px
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select
from xgboost import XGBRegressor

from app.database import create_db_and_tables, engine
from app.models import Revenue, RevenueCreate, RevenueRead, RevenueUpdate

from sklearn.ensemble import RandomForestRegressor

import plotly.graph_objects as go

from joblib import dump, load
from datetime import date

# xgb = XGBRegressor()
# xgb.load_model("bin/xgb_2023-12-08_rev1.json")

features = ["day_of_month", "day_of_week", "day_of_year", "month"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    # load_csv_to_db("swic-revenue-rev1.csv")
    retrain_model()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")


def sqlmodel_to_df(objs: list[SQLModel]) -> pd.DataFrame:
    """Convert a SQLModel objects into a pandas DataFrame."""
    records = [i.model_dump() for i in objs]
    df = pd.DataFrame.from_records(records)
    return df


def df_to_sqlmodel(df: pd.DataFrame) -> List[SQLModel]:
    """Convert a pandas DataFrame into a a list of SQLModel objects."""
    objs = [Revenue(**row) for row in df.to_dict("records")]
    return objs


def load_csv_to_db(filepath: str):
    df = pd.read_csv(filepath, sep=";")
    df["date"] = pd.to_datetime(df["date"], format=r"%d/%m/%Y")
    objs = df_to_sqlmodel(df)
    with Session(engine) as session:
        for obj in objs:
            session.add(obj)
            session.commit()
            session.refresh(obj)
    print("done")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by="date")

    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


def retrain_model():
    print("retraining model...")
    df = sqlmodel_to_df(get_revenues())
    df["date"] = pd.to_datetime(df["date"])
    df.drop(["id"], inplace=True, axis=1)
    df.sort_values(by="date", ascending=True, inplace=True)

    preprocessed_df = preprocess_df(df)

    X_train = preprocessed_df[features]
    y_train = preprocessed_df["revenue"]

    rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
    dump(rf, "bin/model.joblib")
    print("model retrained!")


@app.get("/")
def index(request: Request):
    data = json.dumps(
        {
            "labels": ["Red", "Blue", "Yellow", "Green", "Purple", "Orange"],
            "datasets": [
                {"label": "# of Votes", "data": [12, 19, 3, 5, 2, 3], "borderWidth": 1}
            ],
            "options": {"layout": {"padding": 20}},
        },
    )

    return templates.TemplateResponse("index.html", {"request": request, "data": data})


@app.get("/get/revenues", response_model=list[RevenueRead])
def get_revenues():
    with Session(engine) as session:
        revenue = session.exec(select(Revenue)).all()
        if not revenue:
            raise HTTPException(status_code=404, detail="Revenue not found")
        return revenue
    

@app.post("/post/revenue", response_model=RevenueRead)
def post_revenue(revenue: RevenueCreate):
    with Session(engine) as session:
        revenue = Revenue(date=revenue.date, revenue=revenue.revenue)
        session.add(revenue)
        session.commit()
        session.refresh(revenue)

        retrain_model()

        return revenue


@app.patch("/patch/revenue/{id}", response_model=RevenueRead)
def patch_revenue(id: int, revenue: RevenueUpdate):
    with Session(engine) as session:
        db_revenue = session.get(Revenue, id)
        if not db_revenue:
            raise HTTPException(status_code=404, detail="Revenue not found")
        revenue_data = revenue.model_dump(exclude_unset=True)
        print(revenue_data)
        for k, v in revenue_data.items():
            setattr(db_revenue, k, v)
        session.add(db_revenue)
        session.commit()
        session.refresh(db_revenue)

        retrain_model()

        return db_revenue


@app.delete("/delete/revenue/{id}")
def delete_revenue(revenue_id: int):
    with Session(engine) as session:
        revenue = session.get(Revenue, revenue_id)
        if not revenue:
            raise HTTPException(status_code=404, detail="Revenue not found")
        session.delete(revenue)
        session.commit()

        retrain_model()
        return {"success": True}


@app.get("/forecast/revenue")
def forecast_revenue(interval: int):
    model = load("bin/model.joblib")

    df = sqlmodel_to_df(get_revenues())
    df["date"] = pd.to_datetime(df["date"])
    df.drop(["id"], inplace=True, axis=1)
    df.sort_values(by="date", ascending=True, inplace=True)
    df["type"] = "Historical"

    future = pd.DataFrame(
        {
            "date": pd.date_range(
                df["date"].max() + pd.Timedelta(1, unit="D"), periods=interval
            )
        }
    )

    preprocessed_future = preprocess_df(future)

    X_test = preprocessed_future[features]

    future["revenue"] = model.predict(X_test)
    future["type"] = "Forecast"
    sum_forecast_revenue = future["revenue"].sum()

    print(future)
    print(df)

    df = df.tail(30)

    df = pd.concat([df, future], ignore_index=True)

    fig = px.line(
        df,
        x="date",
        y="revenue",
        color="type",
        title="Historical and Forecasted Revenue",
        markers=True,
        line_shape="linear",
    )

    # Convert the plot to JSON
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return {"graph": graph_json, "sum_forecast_revenue": sum_forecast_revenue}


@app.get("/dashboard")
def dashboard(
    request: Request, interval: int = Query(7, description="Forecast interval")
):
    forecast = forecast_revenue(interval=interval)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "graph_json": forecast["graph"],
            "sum_forecast_revenue": forecast["sum_forecast_revenue"],
        },
    )


@app.get("/database")
def dev(request: Request):
    return templates.TemplateResponse(
        "index2.html",
        {
            "request": request,
            "revenues": get_revenues(),
        },
    )
