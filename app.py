from fastapi import FastAPI, Depends, Request, Form, status
from fastapi.responses import RedirectResponse
from starlette.templating import Jinja2Templates
from datetime import datetime
from sqlalchemy.orm import Session

import models
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

from models import StockList, PredictiveModel
from sqlalchemy import Table, select
import matplotlib.pyplot as plt
import base64
from io import BytesIO

templates = Jinja2Templates(directory="templates")

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def plot_to_base64(fig):
    """Convert a Matplotlib figure to a Base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    Pmodel_list = db.query(PredictiveModel).all()
    return templates.TemplateResponse("base.html", {"request": request, "Pmodel_list": Pmodel_list})

@app.post("/add")
def add(request: Request, title: str = Form(...), db: Session = Depends(get_db)):
    new_Pmodel = PredictiveModel(title=title)
    db.add(new_Pmodel)
    db.commit()

    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

@app.get("/update/{Pmodel_id}")
def update(request: Request, Pmodel_id: int, db: Session = Depends(get_db)):
    Pmodel = db.query(PredictiveModel).filter(PredictiveModel.id == Pmodel_id).first()
    Pmodel.complete = not Pmodel.complete
    db.commit()

    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

@app.get("/delete/{Pmodel_id}")
def delete(request: Request, Pmodel_id: int, db: Session = Depends(get_db)):
    Pmodel = db.query(PredictiveModel).filter(PredictiveModel.id == Pmodel_id).first()
    db.delete(Pmodel)
    db.commit()

    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

@app.get("/stock_list")
def stock_list(request: Request, db: Session = Depends(get_db)):
    stocks = db.query(StockList).all()
    return templates.TemplateResponse("stock_list.html", {"request": request, "stocks": stocks})

@app.get("/view_stock/{stock_name}")
def view_stock(stock_name: str, request: Request, db: Session = Depends(get_db)):
    # Retrieve stock info from StockList table
    stock_info = db.query(StockList).filter(StockList.stock_name == stock_name).first()

    # Fetch the dynamically created stock data table
    stock_table = Table(stock_name, StockList.metadata, autoload_with=db.bind)
    query = select(stock_table.c.date, stock_table.c.open, stock_table.c.high, stock_table.c.low, stock_table.c.close, stock_table.c.volume)
    stock_data = db.execute(query).fetchall()

    # Convert stock_data to a DataFrame
    import pandas as pd
    df = pd.DataFrame(stock_data, columns=["date", "open", "high", "low", "close", "volume"])
    df.set_index("date", inplace=True)

    # Generate plots for each metric and convert them to Base64
    plots = {}
    for column in ["open", "high", "low", "close", "volume"]:
        fig, ax = plt.subplots(figsize=(12, 7))
        df[column].plot(ax=ax, title=f"{column.capitalize()} Price" if column != "volume" else "Volume")
        ax.set_xlabel("Date")
        ax.set_ylabel(column.capitalize())
        plots[f"{column}_plot"] = plot_to_base64(fig)
        plt.close(fig)

    # Render the template with the stock information and plots
    return templates.TemplateResponse(
        "view_stock.html",
        {
            "request": request,
            "stock": stock_info,
            "close_plot": plots["close_plot"],
            "open_plot": plots["open_plot"],
            "high_plot": plots["high_plot"],
            "low_plot": plots["low_plot"],
            "volume_plot": plots["volume_plot"]
        }
    )

@app.get("/open_model/{model_id}")
def open_model(model_id: int, request: Request, db: Session = Depends(get_db)):
    model = db.query(PredictiveModel).filter(PredictiveModel.id == model_id).first()
    return templates.TemplateResponse("model_details.html", {"request": request, "model": model})

@app.post("/submit_model_inputs/{model_id}")
def submit_model_inputs(
    model_id: int, 
    trade_size: float = Form(...), 
    target_trade_profit: float = Form(...),
    trade_loss_limit: float = Form(...),
    test_end_date: str = Form(...),
    max_trade_duration: int = Form(...),
    training_duration: int = Form(...),
    test_duration: int = Form(...),
    sma_1_duration: int = Form(...),
    sma_2_duration: int = Form(...),
    sma_3_duration: int = Form(...),
    db: Session = Depends(get_db)
):
    # Convert test_end_date from string to date object
    test_end_date = datetime.strptime(test_end_date, "%Y-%m-%d").date()

    # Update model details directly in PredictiveModels table
    db.query(PredictiveModel).filter(PredictiveModel.id == model_id).update({
        "trade_size": trade_size,
        "target_trade_profit": target_trade_profit,
        "trade_loss_limit": trade_loss_limit,
        "test_end_date": test_end_date,
        "max_trade_duration": max_trade_duration,
        "training_duration": training_duration,
        "test_duration": test_duration,
        "sma_1_duration": sma_1_duration,
        "sma_2_duration": sma_2_duration,
        "sma_3_duration": sma_3_duration
    })
    db.commit()
    return RedirectResponse(url=app.url_path_for("home"), status_code=status.HTTP_303_SEE_OTHER)

