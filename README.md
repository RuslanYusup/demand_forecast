Проект по предсказанию спроса на нефтехимические продукты (полиэтилен, полипропилен, мэг, полистирол, термопласты.

Цель - предсказание спроса по контрактному продукту в определенный день.

DATA: Данные были получены из открытого набора датасета ПАО "Сибур" за 2018-2019 гг. Данные содержат информацию
о продажах полимеров в разрезе стран, видов контракта, кодам менеджера и пр. Изначально данные не разделены на тестовую
и обучающую выборку.

Поля в наборе данных:

material_code - код продукта,

company_code - код клиента, который всегда равен 0 для спотовых сделок,

country - страна, в которую осуществляется продажа,

region - регион внутри страны, в которую осуществляется продажа; для большинства стран не детализирован,

manager_code - код менеджера, ведущего сделку,

month - месяц сделки,

material_lvl1_name, material_lvl2_name, material_lvl3_name - группировка продукта по категориям разных уровней,

contract_type - тип сделки,

date - точная дата сделки,

volume - объем сделки.

Код вычисляет предсказания, 
т. е. суммарный объем для следующего месяца для каждой группы 
material_code, company_code, country, region, manager_code.

в наборе данных есть 941 группа 
material_code, company_code, country, region, manager_code

В ноутбуке main.ipynb проведен первичный EDA и построена базовая модель

В качестве baseline была использована catboost, так как она хорошо работает с временными рядами
и категориальными признаками. Для проверки модели была использована rolling cross-validation модели,
где датасет был разбит на несколько фолдов, следовавших друг за другом. В конце усреднили
оценки качества модели.


# периоды тренировочной, валидационной и тестовой выборок - 92 тыс. строк
train = "2019-01-01", "2019-10-01"
valid= "2019-11-01", "2020-02-01"
test= "2020-03-01", "2020-07-01"

Для оценки качества была использована метрика RMSLE. Выбор метрики был обусловлен тем, что она
учитывает относительную, а не абсолютную ошибку, т.е она штрафует модель больше за неправильные
предсказания на маленьких значениях спроса, чем на больших. Это нужно, так как имеется большой
разброс в значениях спроса, также она устойчива к выбросам, которые могут исказить среднее значение
ошибки.
Также RMSLE имеет естественную интерпретацию, так как измеряется по логирифмической шкале.
Вместе с тем:
- метрика не определена если фактический или предстказанный спрос отрицательный (проверка
неотрицательных значений)
- метрика не симметричка, т.е. она штрафует модель больше за недопредсказание, чем за перепредсказание
- она не учитывает важность разных продуктов (в смысле маржи), тут можно было бы обогатить
датасет в зависимости от важности продукта в общей выручке.


Пайплайн с моделью состоит из трех основных элементов

make_dataset: чтения данных
features/build_transformers: обработки признаков, в которую входят
обработка и создание фичей: временные признаки
model_fit_predict: обучаем классическую модель Catboost 



==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


