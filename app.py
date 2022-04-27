from IPython.display import display
import MLBacktraderBase as models
import datetime

m = models.MLBacktraderBase("AAPL", datetime.datetime(2020,1,1), datetime.datetime(2022,1,1))
m.download_data()
m.create_market_direction()
m.save_price_daily_returns_chart_image(image_to_file=True)
m.create_lags_data()
m.create_bins_data()
m.training_models()
m.models_prediction_position()
m.models_performance_evaluation()
m.models_calculate_total_return_std()
m.number_of_trades()
m.disp_performacne(image_to_file=True)
display(m.get_data())