"""
Projeto Final - Machine Learning Clássico com o Brazilian E-Commerce Dataset (Olist)
Autores: Guilherme Rosa e Miguel Machado

MODELOS DEFINIDOS PARA O PROJETO
--------------------------------
1. Random Forest Classifier -> classifica risco de atraso (0/1)
2. Regressão Linear Múltipla -> estima quantos dias de atraso existem
3. Regressão Polinomial -> tenta capturar relações não lineares na estimativa
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =============================================================================
# 1. CONFIGURAÇÃO
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Se a pasta data/ não existir, o código também aceita os CSVs
# diretamente ao lado do arquivo .py. Isso facilita o primeiro teste.
if not (DATA_DIR / "olist_orders_dataset.csv").exists():
    DATA_DIR = BASE_DIR

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

ORDERS_FILE = DATA_DIR / "olist_orders_dataset.csv"
ITEMS_FILE = DATA_DIR / "olist_order_items_dataset.csv"

# O relatório especifica uma divisão de 70% treino e 30% teste.
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Corte inicial usado somente como fallback. O pipeline calcula depois um
# limiar mais adequado usando as probabilidades out-of-fold do treino.
DEFAULT_RISK_THRESHOLD = 0.50

# Quantidade de folds para gerar a probabilidade out-of-fold do treino.
OOF_SPLITS = 5

# Configuração da busca de hiperparâmetros.
# Um número moderado de tentativas mantém o treinamento viável em notebooks
# e computadores mais modestos, sem eliminar a etapa de otimização.
RF_SEARCH_N_ITER = 3
RF_SEARCH_CV = 2
RF_N_JOBS = 2
# Tamanho máximo da amostra usada somente para a busca de hiperparâmetros.
# O modelo final volta a ser treinado com 100% do conjunto de treino.
RF_TUNING_SAMPLE = 10000


# =============================================================================
# 2. FUNÇÕES AUXILIARES
# =============================================================================

def ensure_output_dirs() -> None:
    """Cria as pastas onde os resultados serão salvos."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os dois datasets brutos usados no projeto."""

    if not ORDERS_FILE.exists():
        raise FileNotFoundError(
            f"Não encontrei {ORDERS_FILE}. "
            "Coloque olist_orders_dataset.csv dentro da pasta data/."
        )

    if not ITEMS_FILE.exists():
        raise FileNotFoundError(
            f"Não encontrei {ITEMS_FILE}. "
            "Coloque olist_order_items_dataset.csv dentro da pasta data/."
        )

    orders = pd.read_csv(ORDERS_FILE)
    items = pd.read_csv(ITEMS_FILE)

    print("\n=== DATASETS BRUTOS ===")
    print(f"orders     : {orders.shape[0]:,} linhas x {orders.shape[1]} colunas")
    print(f"order_items: {items.shape[0]:,} linhas x {items.shape[1]} colunas")

    return orders, items


def validate_required_columns(
    orders: pd.DataFrame,
    items: pd.DataFrame,
) -> None:
    """Confere se as colunas necessárias existem nos arquivos."""

    required_orders = {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    }

    required_items = {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    }

    missing_orders = required_orders - set(orders.columns)
    missing_items = required_items - set(items.columns)

    if missing_orders:
        raise ValueError(
            "Colunas ausentes em olist_orders_dataset.csv: "
            f"{sorted(missing_orders)}"
        )

    if missing_items:
        raise ValueError(
            "Colunas ausentes em olist_order_items_dataset.csv: "
            f"{sorted(missing_items)}"
        )


def get_feature_columns() -> list[str]:
    """Retorna apenas as variáveis que podem ser usadas como entrada do modelo."""

    return [
        "approval_time_hours",
        "item_count",
        "total_price",
        "total_freight",
        "mean_item_price",
        "unique_products",
        "unique_sellers",
        "freight_ratio",
        "estimated_delivery_days",
        "shipping_limit_days",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
    ]


def prepare_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara uma tabela no nível de PEDIDO.

    O dataset de itens possui várias linhas por pedido. Como o problema
    de previsão é definido por order_id, os itens são agregados antes
    do treinamento.
    """

    orders = orders.copy()
    items = items.copy()

    # -------------------------------------------------------------------------
    # 1) Converter strings de data para datetime
    # -------------------------------------------------------------------------

    order_date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in order_date_columns:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    items["shipping_limit_date"] = pd.to_datetime(
        items["shipping_limit_date"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # 2) Manter apenas pedidos efetivamente entregues
    # -------------------------------------------------------------------------

    # Para descobrir se houve atraso precisamos saber a data real da entrega.
    # Portanto, pedidos cancelados, indisponíveis etc. não servem para criar
    # o alvo desta análise.
    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()

    print("\n=== FILTRO DE PEDIDOS ENTREGUES ===")
    print(f"Pedidos entregues: {len(delivered):,}")

    # Essas datas são indispensáveis para construir o target.
    required_target_dates = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    before = len(delivered)
    delivered = delivered.dropna(subset=required_target_dates)
    print(f"Após remover datas essenciais ausentes: {len(delivered):,}")
    print(f"Registros removidos nessa etapa: {before - len(delivered):,}")

    # -------------------------------------------------------------------------
    # 3) Agregar order_items para o nível de pedido
    # -------------------------------------------------------------------------

    # Um pedido pode ter vários itens. Essas agregações transformam isso em
    # uma linha por order_id.
    items_agg = (
        items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            mean_item_price=("price", "mean"),
            unique_products=("product_id", "nunique"),
            unique_sellers=("seller_id", "nunique"),
            # Usamos a última data de limite de envio entre os itens do pedido.
            shipping_limit_date=("shipping_limit_date", "max"),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # 4) Fazer o JOIN pelo order_id
    # -------------------------------------------------------------------------

    df = delivered.merge(
        items_agg,
        on="order_id",
        how="inner",
        validate="one_to_one",
    )

    print(f"Após cruzar orders + order_items: {len(df):,} pedidos")

    # -------------------------------------------------------------------------
    # 5) Criar features de tempo
    # -------------------------------------------------------------------------

    # Quanto tempo demorou para o pagamento/pedido ser aprovado.
    df["approval_time_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    # Prazo estimado pelo e-commerce, contado desde a compra.
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Limite dado para o vendedor enviar o pedido.
    df["shipping_limit_days"] = (
        df["shipping_limit_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Proporção do frete em relação ao valor dos produtos.
    # Exemplo: frete 20 / produtos 100 = 0.20 (20%).
    df["freight_ratio"] = np.where(
        df["total_price"] > 0,
        df["total_freight"] / df["total_price"],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # 6) Criar variáveis de calendário
    # -------------------------------------------------------------------------

    purchase = df["order_purchase_timestamp"]
    df["purchase_month"] = purchase.dt.month
    df["purchase_dayofweek"] = purchase.dt.dayofweek
    df["purchase_hour"] = purchase.dt.hour

    # -------------------------------------------------------------------------
    # 7) Criar o TARGET de atraso
    # -------------------------------------------------------------------------

    # O documento define atraso como o excesso em relação à data estimada.
    # Normalizamos para a data, pois a estimativa da Olist aparece como data
    # de referência, não como horário exato de entrega.
    delivered_date = df["order_delivered_customer_date"].dt.normalize()
    estimated_date = df["order_estimated_delivery_date"].dt.normalize()

    df["days_atraso_raw"] = (
        delivered_date - estimated_date
    ).dt.total_seconds() / 86400

    # Um atraso negativo significa que a entrega ocorreu antes do prazo.
    # Para a regressão de "dias de atraso", usamos zero nesses casos.
    df["days_atraso"] = df["days_atraso_raw"].clip(lower=0)

    # Target binário da classificação.
    df["atrasado"] = (df["days_atraso"] > 0).astype(int)

    # -------------------------------------------------------------------------
    # 8) Tratamento final de valores inválidos
    # -------------------------------------------------------------------------

    # Infinitos podem surgir, por exemplo, numa divisão por zero.
    df = df.replace([np.inf, -np.inf], np.nan)

    features = get_feature_columns()

    before = len(df)
    df = df.dropna(subset=features + ["atrasado", "days_atraso"]).copy()
    removed = before - len(df)

    print(f"Linhas removidas por NaN/inf nas features: {removed:,}")

    return df


# =============================================================================
# 3. EDA - ANÁLISE EXPLORATÓRIA
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """Gera diagnósticos e gráficos da base analítica."""

    print("\n" + "=" * 75)
    print("EDA - ANÁLISE EXPLORATÓRIA DE DADOS")
    print("=" * 75)

    print("\n--- Tamanho da base ---")
    print(f"Linhas: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")

    print("\n--- Nulos ---")
    nulls = df.isna().sum().sort_values(ascending=False)
    print(nulls[nulls > 0] if (nulls > 0).any() else "Nenhum nulo nas colunas da base analítica.")

    print("\n--- Distribuição do target ---")
    target_table = pd.DataFrame({
        "quantidade": df["atrasado"].value_counts().sort_index(),
        "percentual": df["atrasado"].value_counts(normalize=True).sort_index() * 100,
    })
    print(target_table.round(2))

    print("\n--- Estatísticas dos dias de atraso ---")
    print(
        df["days_atraso"].describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
        ).to_string()
    )

    print("\n--- Possíveis outliers no alvo ---")
    for threshold in [30, 60, 90, 120]:
        count = int((df["days_atraso"] > threshold).sum())
        percentage = count / len(df) * 100
        print(f"> {threshold:>3} dias: {count:>4} registros ({percentage:.4f}%)")

    # Exportamos os casos mais extremos para inspeção manual.
    # Eles NÃO são removidos automaticamente. Isso é importante para que
    # o relatório consiga justificar qualquer limpeza posterior.
    possible_anomalies = df.loc[
        (df["days_atraso"] > 60)
        | (df["shipping_limit_days"] > 60)
        | (df["approval_time_hours"] > 168),
        [
            "order_id",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "shipping_limit_date",
            "days_atraso",
            "shipping_limit_days",
            "approval_time_hours",
        ],
    ].sort_values("days_atraso", ascending=False)

    possible_anomalies.to_csv(
        TABLES_DIR / "possiveis_anomalias_para_inspecao.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Gráfico 1 - distribuição dos dias de atraso
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.hist(
        df["days_atraso"],
        bins=50,
        edgecolor="black",
    )
    plt.title("Distribuição dos Dias de Atraso")
    plt.xlabel("Dias de atraso")
    plt.ylabel("Quantidade de pedidos")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_distribuicao_dias_atraso.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfico 2 - classes
    # -------------------------------------------------------------------------

    counts = df["atrasado"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(
        ["No prazo", "Atrasado"],
        [counts.get(0, 0), counts.get(1, 0)],
    )
    plt.title("Distribuição do Risco de Atraso")
    plt.ylabel("Quantidade de pedidos")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_distribuicao_classes.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfico 3 - prazo estimado x atraso
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.scatter(
        df["estimated_delivery_days"],
        df["days_atraso"],
        alpha=0.25,
        s=10,
    )
    plt.title("Prazo Estimado x Dias de Atraso")
    plt.xlabel("Prazo estimado (dias)")
    plt.ylabel("Dias de atraso")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_prazo_estimado_vs_atraso.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfico 4 - distribuição das principais variáveis numéricas
    # -------------------------------------------------------------------------

    for feature in [
        "total_price",
        "total_freight",
        "approval_time_hours",
        "estimated_delivery_days",
    ]:
        plt.figure(figsize=(9, 5))
        plt.hist(df[feature], bins=50, edgecolor="black")
        plt.title(f"Distribuição - {feature}")
        plt.xlabel(feature)
        plt.ylabel("Quantidade")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"feature_{feature}.png", dpi=150)
        plt.close()

    print(f"\nGráficos salvos em: {FIGURES_DIR}")


# =============================================================================
# 4. DIVISÃO TREINO/TESTE
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Divide a base em 70% treino e 30% teste.

    O stratify mantém aproximadamente a mesma proporção de pedidos atrasados
    e no prazo nos dois conjuntos.
    """

    features = get_feature_columns()

    X = df[features].copy()
    y_class = df["atrasado"].copy()
    y_reg = df["days_atraso"].copy()

    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )

    print("\n=== DIVISÃO 70/30 ===")
    print(f"Treino: {len(X_train):,}")
    print(f"Teste : {len(X_test):,}")
    print("\nProporção de atraso no treino:")
    print(y_class_train.mean())
    print("Proporção de atraso no teste:")
    print(y_class_test.mean())

    return (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    )


# =============================================================================
# 5. RANDOM FOREST - CLASSIFICAÇÃO
# =============================================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Treina e otimiza o Random Forest.

    Para manter o treinamento viável em computadores modestos, a busca de
    hiperparâmetros pode usar uma amostra estratificada do conjunto de treino.
    Depois de descobrir os melhores parâmetros, o modelo FINAL é treinado com
    todas as linhas disponíveis no treinamento.

    O scoring escolhido para a busca é Recall porque, neste problema, perder
    um pedido que realmente atrasaria é um falso negativo relevante.
    """

    # -------------------------------------------------------------------------
    # 1) Configuração da busca
    # -------------------------------------------------------------------------

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        class_weight="balanced",
    )

    param_distributions = {
        "n_estimators": [50, 75, 100],
        "max_depth": [None, 15, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    # -------------------------------------------------------------------------
    # 2) Amostra estratificada para acelerar a busca
    # -------------------------------------------------------------------------

    if RF_TUNING_SAMPLE is not None and len(X_train) > RF_TUNING_SAMPLE:
        X_search, _, y_search, _ = train_test_split(
            X_train,
            y_train,
            train_size=RF_TUNING_SAMPLE,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
        print(
            f"\nBusca de hiperparâmetros usando amostra estratificada de "
            f"{len(X_search):,} linhas."
        )
    else:
        X_search = X_train
        y_search = y_train
        print("\nBusca de hiperparâmetros usando todo o conjunto de treino.")

    # -------------------------------------------------------------------------
    # 3) RandomizedSearchCV
    # -------------------------------------------------------------------------

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=RF_SEARCH_N_ITER,
        scoring="recall",
        cv=RF_SEARCH_CV,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        verbose=1,
        refit=False,
    )

    search.fit(X_search, y_search)

    # Guardamos todas as combinações testadas para documentar a otimização
    # no relatório final.
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        TABLES_DIR / "random_forest_cv_results.csv",
        index=False,
    )

    print("\n=== RANDOM FOREST OTIMIZADO ===")
    print("Melhores parâmetros:")
    print(search.best_params_)
    print(f"Melhor Recall médio na validação: {search.best_score_:.4f}")

    # -------------------------------------------------------------------------
    # 4) Modelo final treinado com 100% do treino
    # -------------------------------------------------------------------------

    final_model = RandomForestClassifier(
        **search.best_params_,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        class_weight="balanced",
    )

    print("Treinando modelo final com todo o conjunto de treino...")
    final_model.fit(X_train, y_train)

    return final_model, search


# =============================================================================
# 6. AVALIAÇÃO DA CLASSIFICAÇÃO
# =============================================================================

def evaluate_random_forest(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    risk_threshold: float,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Calcula as métricas e gera gráficos do classificador."""

    if y_prob is None:
        y_prob = model.predict_proba(X_test)[:, 1]

    # Em vez de usar automaticamente 0,50, aplicamos o limiar escolhido
    # exclusivamente com dados de treinamento/validação.
    y_pred = (y_prob >= risk_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        # PR-AUC é útil quando uma classe é muito menos frequente que a outra.
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        TABLES_DIR / "metricas_random_forest.csv",
        index=False,
    )

    print("\n=== MÉTRICAS - RANDOM FOREST ===")
    print(f"Limiar de classificação utilizado: {risk_threshold:.4f}")
    print(metrics_df.round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    # Matriz de confusão
    # -------------------------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No prazo", "Atrasado"],
    ).plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Matriz de Confusão - Random Forest")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_matriz_confusao_random_forest.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Curva ROC
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax,
        name="Random Forest",
    )
    ax.set_title("Curva ROC - Random Forest")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_curva_roc_random_forest.png", dpi=150)
    plt.close(fig)

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }


# =============================================================================
# 7. PROBABILIDADE OOF PARA A REGRESSÃO
# =============================================================================

def create_oof_risk_probability(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
):
    """
    Cria a probabilidade de atraso que será usada na regressão.

    POR QUE NÃO USAR model.predict_proba(X_train) DIRETAMENTE?
    -----------------------------------------------------------
    Porque o modelo já teria visto aquelas mesmas linhas durante o treino.
    Isso pode produzir uma probabilidade otimista demais.

    Com cross_val_predict, cada linha de treino recebe uma probabilidade
    produzida por um modelo que não treinou naquela linha.
    """

    cv = StratifiedKFold(
        n_splits=OOF_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_probability = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=RF_N_JOBS,
    )[:, 1]

    # Depois do OOF, treinamos um classificador final com 100% do treino.
    model.fit(X_train, y_train)
    test_probability = model.predict_proba(X_test)[:, 1]

    # Escolhemos o limiar no conjunto de treino usando as previsões OOF.
    # Assim, o teste continua intocado até a avaliação final.
    precision, recall, thresholds = precision_recall_curve(
        y_train,
        train_probability,
    )

    # precision/recall têm uma posição a mais que thresholds.
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )

    if len(thresholds) == 0:
        risk_threshold = DEFAULT_RISK_THRESHOLD
    else:
        best_index = int(np.nanargmax(f1_scores))
        risk_threshold = float(thresholds[best_index])

    threshold_df = pd.DataFrame({
        "metric": ["risk_threshold", "default_threshold"],
        "value": [risk_threshold, DEFAULT_RISK_THRESHOLD],
    })
    threshold_df.to_csv(
        TABLES_DIR / "limiar_risco.csv",
        index=False,
    )

    print(f"\nLimiar escolhido a partir do treino (F1 máximo): {risk_threshold:.4f}")

    return train_probability, test_probability, risk_threshold


# =============================================================================
# 8. REGRESSÃO LINEAR MÚLTIPLA E POLINOMIAL
# =============================================================================

def build_multiple_linear_pipeline() -> Pipeline:
    """
    Regressão Linear Múltipla.

    Forma conceitual:
        y = b0 + b1*x1 + b2*x2 + ... + bn*xn

    O StandardScaler deixa as variáveis em escalas comparáveis antes
    do ajuste do modelo.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_polynomial_linear_pipeline() -> Pipeline:
    """
    Regressão Polinomial de grau 2.

    A Regressão Linear continua sendo o estimador final, mas recebe
    novas features geradas por PolynomialFeatures:
        x1^2, x2^2, x1*x2, ...

    Isso permite representar relações não lineares.
    """

    return Pipeline(
        steps=[
            ("scaler_input", StandardScaler()),
            (
                "poly",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
            ),
            ("scaler_poly", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def prepare_regression_data(
    X_train: pd.DataFrame,
    y_class_train: pd.Series,
    y_reg_train: pd.Series,
    risk_probability_train: np.ndarray,
):
    """
    Prepara a base para a regressão.

    A regressão responde:
        "Entre os pedidos atrasados, ocorreram quantos dias de atraso?"

    Então treinamos a regressão somente nas linhas em que o target
    real indica atraso.
    """

    X_reg_train = X_train.copy()
    X_reg_train["risk_probability"] = risk_probability_train

    delayed_mask = y_class_train.eq(1)

    X_reg_train = X_reg_train.loc[delayed_mask].copy()
    y_reg_train = y_reg_train.loc[delayed_mask].copy()

    return X_reg_train, y_reg_train


def train_regression_models(
    X_reg_train: pd.DataFrame,
    y_reg_train: pd.Series,
):
    """Treina os dois modelos de regressão do projeto."""

    print("\n=== TREINAMENTO DA REGRESSÃO ===")
    print(f"Pedidos atrasados utilizados: {len(X_reg_train):,}")

    models = {
        "MultipleLinearRegression": build_multiple_linear_pipeline(),
        "PolynomialRegression": build_polynomial_linear_pipeline(),
    }

    for name, model in models.items():
        print(f"Treinando: {name}")
        model.fit(X_reg_train, y_reg_train)

    return models


# =============================================================================
# 9. AVALIAÇÃO DA REGRESSÃO
# =============================================================================

def evaluate_regression_models(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_class_test: pd.Series,
    y_reg_test: pd.Series,
    risk_probability_test: np.ndarray,
    risk_threshold: float,
) -> pd.DataFrame:
    """
    Avalia:

    1) Severidade do atraso:
       somente pedidos que realmente atrasaram.

    2) Pipeline completo:
       primeiro o Random Forest decide o risco; depois a regressão estima
       a quantidade de dias.
    """

    X_reg_test = X_test.copy()
    X_reg_test["risk_probability"] = risk_probability_test

    delayed_mask = y_class_test.eq(1)

    X_test_delayed = X_reg_test.loc[delayed_mask].copy()
    y_test_delayed = y_reg_test.loc[delayed_mask].copy()

    results = []

    for name, model in models.items():
        # ---------------------------------------------------------------------
        # A) Modelo avaliado somente nos pedidos que realmente atrasaram.
        # ---------------------------------------------------------------------

        pred_delayed = model.predict(X_test_delayed)
        pred_delayed = np.clip(pred_delayed, 0, None)

        mae_delayed = mean_absolute_error(
            y_test_delayed,
            pred_delayed,
        )

        rmse_delayed = np.sqrt(
            mean_squared_error(
                y_test_delayed,
                pred_delayed,
            )
        )

        r2_delayed = r2_score(
            y_test_delayed,
            pred_delayed,
        )

        # ---------------------------------------------------------------------
        # B) Pipeline completo: classificação + regressão.
        # ---------------------------------------------------------------------

        pred_all = model.predict(X_reg_test)
        pred_all = np.clip(pred_all, 0, None)

        # Só usa a previsão da regressão se o Random Forest indicar risco.
        final_prediction = np.where(
            risk_probability_test >= risk_threshold,
            pred_all,
            0,
        )

        mae_pipeline = mean_absolute_error(
            y_reg_test,
            final_prediction,
        )

        rmse_pipeline = np.sqrt(
            mean_squared_error(
                y_reg_test,
                final_prediction,
            )
        )

        r2_pipeline = r2_score(
            y_reg_test,
            final_prediction,
        )

        results.append({
            "modelo": name,
            "mae_atrasados": mae_delayed,
            "rmse_atrasados": rmse_delayed,
            "r2_atrasados": r2_delayed,
            "mae_pipeline_completo": mae_pipeline,
            "rmse_pipeline_completo": rmse_pipeline,
            "r2_pipeline_completo": r2_pipeline,
        })

        # Gráfico de valores reais x previstos nos atrasados.
        plt.figure(figsize=(7, 6))
        plt.scatter(
            y_test_delayed,
            pred_delayed,
            alpha=0.25,
            s=12,
        )
        min_value = min(y_test_delayed.min(), pred_delayed.min())
        max_value = max(y_test_delayed.max(), pred_delayed.max())
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
        )
        plt.title(f"Real x Predito - {name}")
        plt.xlabel("Dias de atraso reais")
        plt.ylabel("Dias de atraso previstos")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"regressao_real_vs_predito_{name}.png",
            dpi=150,
        )
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        TABLES_DIR / "metricas_regressao.csv",
        index=False,
    )

    print("\n=== MÉTRICAS DE REGRESSÃO ===")
    print(results_df.round(4).to_string(index=False))

    return results_df


# =============================================================================
# 10. IMPORTÂNCIA DAS FEATURES
# =============================================================================

def save_feature_importance(
    rf_model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extrai a importância das features do Random Forest.

    Isso ajudará na análise do projeto para responder:
        "Quais variáveis tiveram maior importância para a classificação?"
    """

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        TABLES_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    top = importance_df.head(10).sort_values("importance")

    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 10 - Importância das Features no Random Forest")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_importancia_features_random_forest.png", dpi=150)
    plt.close()

    print("\n=== IMPORTÂNCIA DAS FEATURES ===")
    print(importance_df.head(10).round(4).to_string(index=False))

    return importance_df


# =============================================================================
# 11. MAIN
# =============================================================================

def main() -> None:
    """Executa o pipeline completo."""

    ensure_output_dirs()

    # -------------------------------------------------------------------------
    # A) Carregamento
    # -------------------------------------------------------------------------

    orders, items = load_data()
    validate_required_columns(orders, items)

    # -------------------------------------------------------------------------
    # B) Preparação + feature engineering + targets
    # -------------------------------------------------------------------------

    df = prepare_dataset(orders, items)

    # Salvamos uma cópia para facilitar a inspeção durante o desenvolvimento.
    df.to_csv(
        TABLES_DIR / "dataset_analitico_nivel_pedido.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # C) EDA
    # -------------------------------------------------------------------------

    run_eda(df)

    # -------------------------------------------------------------------------
    # D) Split 70/30
    # -------------------------------------------------------------------------

    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    ) = split_data(df)

    # -------------------------------------------------------------------------
    # E) Random Forest
    # -------------------------------------------------------------------------

    rf_model, rf_search = train_random_forest(
        X_train,
        y_class_train,
    )

    # -------------------------------------------------------------------------
    # F) Probabilidade de risco e escolha do limiar
    # -------------------------------------------------------------------------

    (
        risk_probability_train,
        risk_probability_test,
        risk_threshold,
    ) = create_oof_risk_probability(
        rf_model,
        X_train,
        y_class_train,
        X_test,
    )

    # Agora vai avaliar o teste usando o limiar que foi definido sem tocar
    # nos dados de teste.
    classification_output = evaluate_random_forest(
        rf_model,
        X_test,
        y_class_test,
        risk_threshold=risk_threshold,
        y_prob=risk_probability_test,
    )

    # -------------------------------------------------------------------------
    # G) Base de regressão
    # -------------------------------------------------------------------------

    X_reg_train, y_reg_train_delayed = prepare_regression_data(
        X_train,
        y_class_train,
        y_reg_train,
        risk_probability_train,
    )

    regression_models = train_regression_models(
        X_reg_train,
        y_reg_train_delayed,
    )

    # -------------------------------------------------------------------------
    # H) Avaliação das regressões
    # -------------------------------------------------------------------------

    regression_results = evaluate_regression_models(
        regression_models,
        X_test,
        y_class_test,
        y_reg_test,
        risk_probability_test,
        risk_threshold,
    )

    # -------------------------------------------------------------------------
    # I) Importância das features
    # -------------------------------------------------------------------------

    feature_importance = save_feature_importance(
        rf_model,
        get_feature_columns(),
    )

    # -------------------------------------------------------------------------
    # J) Relatório resumido em texto
    # -------------------------------------------------------------------------

    summary = [
        "RESULTADO DO PIPELINE OLIST",
        "===========================",
        "",
        f"Pedidos utilizados: {len(df):,}",
        f"Pedidos atrasados: {int(df['atrasado'].sum()):,}",
        f"Taxa de atraso: {df['atrasado'].mean() * 100:.2f}%",
        "",
        "RANDOM FOREST",
        f"Accuracy:  {classification_output['metrics']['accuracy']:.4f}",
        f"Precision: {classification_output['metrics']['precision']:.4f}",
        f"Recall:    {classification_output['metrics']['recall']:.4f}",
        f"F1:        {classification_output['metrics']['f1']:.4f}",
        f"ROC-AUC:   {classification_output['metrics']['roc_auc']:.4f}",
        f"PR-AUC:    {classification_output['metrics']['pr_auc']:.4f}",
        f"Limiar:    {risk_threshold:.4f}",
        "",
        "REGRESSÃO",
        regression_results.round(4).to_string(index=False),
        "",
        "Observação: os resultados acima são baseados nos datasets fornecidos.",
    ]

    (OUTPUT_DIR / "resumo_resultados.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("PIPELINE FINALIZADO")
    print("=" * 75)
    print(f"Resultados: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
