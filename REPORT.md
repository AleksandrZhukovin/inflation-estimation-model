# Звіт

**Основні результати:**

| Модель | RMSE (UA, 2020–2021) |
|:---|---:|
| Чистий ARIMA | 5.7633 |
| Гібрид ARIMA+XGBoost | 5.4136 |
| Чистий XGBoost | 3.8887 |

Гібридна модель перевершує ARIMA на 6.1%. Чистий XGBoost показує найкращий результат на 24-місячному горизонті.

---

## Дані

**Файл:** `data/diploma_dataset_1.csv`
**Частота:** місячна
**Цільова змінна:** `CPI_Index` (рік-до-року, %)

### Часовий діапазон

| Країна | Початок | Кінець | Записів |
|:---|:---|:---|---:|
| UA | 2007-01-31 | 2025-12-31 | ~228 |
| LT | 2004-01-31 | 2025-12-31 | 264 |
| LV | 2004-01-31 | 2025-12-31 | 264 |

### Предиктори

| Ознака | Опис |
|:---|:---|
| `Official_Currency_to_USD` | Офіційний курс до USD |
| `Official_Currency_to_EUR` | Офіційний курс до EUR |
| `Key_Rate` | Ключова процентна ставка (%) |
| `Brent_Oil` | Ціна нафти Brent (USD/бар) |
| `GDP_yoy` | ВВП рік-до-року (%) |
| `Money_Supply_yoy` | Грошова маса М2 рік-до-року (%) |
| `Global_GPR_Index` | Глобальний індекс геополітичного ризику |
| `Gold_Price` | Ціна золота (USD/трой. унція) |
| `VIX_Index` | Індекс волатильності CBOE |
| `Unemployment` | Рівень безробіття (%) |
| `REER` | Реальний ефективний обмінний курс |
| `Metals_Index` | Індекс металів |
| `FAO_FFPI` | Індекс продовольчих цін ФАО |
| `Industrial_Production_yoy` | Промислове виробництво рік-до-року (%) |

---

## Гіперпараметрів

**Метод:** Optuna TPE, 150 випробувань
**Найкращий CV RMSE:** 2.9176

### Найкращі гіперпараметри

| Параметр | Значення |
|:---|---:|
| `eta` | 0.0473 |
| `max_depth` | 7 |
| `min_child_weight` | 10 |
| `gamma` | 0.0421 |
| `reg_alpha` | 0.3010 |
| `reg_lambda` | 1.4333 |
| `subsample` | 0.7584 |
| `colsample_bytree` | 0.8127 |

---

## Порівняння моделей

## Горизонтний аналіз

| Горизонт (місяців) | Гібрид | Чистий ARIMA | Чистий XGBoost |
|---:|---:|---:|---:|
| 1 | **0.1039** | 0.1053 | 3.7666 |
| 3 | 0.4230 | **0.2928** | 4.8146 |
| 6 | 0.3913 | **0.3024** | 5.3962 |
| 12 | **0.8486** | 0.8516 | 5.1634 |

На короткому горизонті (h=1) гібридна модель незначно перевершує ARIMA. На середніх горизонтах (h=3, 6) ARIMA точніша — XGBoost-корекція залишків дещо погіршує прогноз. На горизонті h=12 гібрид і ARIMA практично рівнозначні.

---

# Топ-10 ознак за середнім SHAP

| Ознака | Mean SHAP | Інтерпретація |
|:---|---:|:---|
| `Year_trend` | 0.4997 | Довгостроковий лінійний тренд — домінуюча ознака |
| `FAO_FFPI` | 0.1110 | Глобальні продовольчі ціни |
| `Industrial_Production_yoy` | 0.1068 | Внутрішній виробничий цикл |
| `Brent_Oil` | 0.1035 | Ціна на енергоносії |
| `REER` | 0.1030 | Реальний обмінний курс |
| `Official_Currency_to_USD` | 0.1007 | Номінальний обмінний курс USD |
| `Official_Currency_to_EUR` | 0.0970 | Номінальний обмінний курс EUR |
| `GDP_yoy` | 0.0875 | Темп економічного зростання |
| `Key_Rate` | 0.0817 | Монетарна політика |
| `VIX_Index` | 0.0816 | Глобальна ринкова невизначеність |
| `Global_GPR_Index` | 0.0752 | Геополітичний ризик |

---

## Додаткова інформація

**Файли моделей:**
- `outputs/models/arima_ua.pkl` — ARIMA UA
- `outputs/models/arima_lt.pkl` — ARIMA LT
- `outputs/models/arima_lv.pkl` — ARIMA LV
- `outputs/models/xgboost_final.pkl` — фінальна гібридна XGBoost-модель
- `outputs/models/xgboost_pure.pkl` — бенчмарк XGBoost (прямий)
- `outputs/models/best_params.json` — найкращі гіперпараметри

**Ключові таблиці результатів:**
- `outputs/tables/arima/arima_specifications.csv`
- `outputs/tables/evaluation/model_comparison.csv`
- `outputs/tables/evaluation/dm_test_pvalues.csv`
- `outputs/tables/horizon/horizon_metrics.csv`
- `outputs/tables/shap/global_feature_ranking.csv`
- `outputs/tables/tuning/hyperparameter_search_top10.csv`
- `outputs/tables/xgboost/training_summary.csv`
