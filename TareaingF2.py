#Tarea 2 ingenieria financiera.
import pandas as pd
import numpy as np

# Lee el archivo Excel
ruta_archivo = "Base de datos completa.xlsx"  # Cambia la ruta si el archivo no está en el mismo directorio
# Cargar todas las hojas del archivo Excel
datos = pd.read_excel(ruta_archivo, sheet_name=None)

# Iterar sobre cada hoja y mostrar las primeras filas
for hoja, df in datos.items():
    print(f"Hoja: {hoja}")
    print(df.head())

# --- Inicio del análisis para la Tarea ---

# Convertir 'Date' a datetime y calcular retornos logarítmicos diarios excluyendo el último año.
log_returns = {}
for asset, df in datos.items():
    # Procesar solo si existen columnas 'Date' y 'Price' y el DataFrame no está vacío.
    if df.empty or 'Date' not in df.columns or 'Price' not in df.columns:
        print(f"Sheet {asset} skipped, missing 'Date' or 'Price' columns or is empty.")
        continue
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    cutoff = df['Date'].max() - pd.DateOffset(years=1)
    df_filtered = df[df['Date'] <= cutoff].copy()
    df_filtered['log_return'] = np.log(df_filtered['Price'] / df_filtered['Price'].shift(1))
    log_returns[asset] = df_filtered.dropna(subset=['log_return'])

# Calcular retornos y volatilidades anuales para cada activo.
expected_returns_annual = {}
volatility_annual = {}
for asset, df in log_returns.items():
    mu_daily = df['log_return'].mean()
    sigma_daily = df['log_return'].std()
    expected_returns_annual[asset] = mu_daily * 252
    volatility_annual[asset] = sigma_daily * np.sqrt(252)

# Unir los retornos en un DataFrame para alinear series.
returns_df = pd.DataFrame({asset: df.set_index('Date')['log_return'] for asset, df in log_returns.items()})

# Calcular matriz de covarianza anualizada.
cov_daily = returns_df.cov()
cov_annual = cov_daily * 252

# a) Imprimir vector de retornos esperados y volatilidades anuales.
print("Retornos esperados anuales por activo:", expected_returns_annual)
print("Volatilidades anuales por activo:", volatility_annual)

# b) Estimar la cartera de mínima varianza (ejemplo de la frontera eficiente).
ones = np.ones(len(returns_df.columns))
inv_cov = np.linalg.inv(cov_annual)
w_min = inv_cov.dot(ones) / ones.dot(inv_cov).dot(ones)
port_return_min = sum(w_min[i] * expected_returns_annual[asset] for i, asset in enumerate(returns_df.columns))
port_vol_min = np.sqrt(w_min.T.dot(cov_annual).dot(w_min))
print("Cartera de mínima varianza:")
print("Pesos:", w_min)
print("Retorno anual esperado:", port_return_min)
print("Volatilidad anual:", port_vol_min)

# c) Recomendación: utilizar la cartera de mínimo riesgo para invertir por 1 año.
print("Recomendación de inversión (cartera de mínimo riesgo):")
print("Pesos:", w_min)
print("Rentabilidad esperada anual:", port_return_min)
print("Volatilidad anual:", port_vol_min)

# d) Considerar una tasa libre de riesgo (por ejemplo, 3%) para obtener la cartera tangente.
risk_free_rate = 0.03
excess_returns = np.array([expected_returns_annual[asset] - risk_free_rate for asset in returns_df.columns])
w_tan = inv_cov.dot(excess_returns) / ones.dot(inv_cov).dot(excess_returns)
tan_return = sum(w_tan[i] * expected_returns_annual[asset] for i, asset in enumerate(returns_df.columns))
tan_vol = np.sqrt(w_tan.T.dot(cov_annual).dot(w_tan))
print("Cartera tangente (nueva frontera eficiente):")
print("Pesos:", w_tan)
print("Retorno anual esperado:", tan_return)
print("Volatilidad anual:", tan_vol)
# --- Fin del análisis ---


# --- Sección e): Cartera de mercado y betas ---
# Suponemos que la cartera de mercado es la cartera tangente calculada en d)
market_weights = w_tan
market_return_daily = returns_df.dot(market_weights)
market_return_expected = market_return_daily.mean() * 252
market_vol_daily = market_return_daily.std()
market_vol_annual = market_vol_daily * np.sqrt(252)
print("Cartera de mercado (tangente):")
print("Pesos:", market_weights)
print("Retorno anual esperado:", market_return_expected)
print("Volatilidad anual:", market_vol_annual)

# Calcular betas de cada activo
market_var_daily = market_return_daily.var()
betas = {}
for asset in returns_df.columns:
    cov_asset = np.cov(returns_df[asset], market_return_daily)[0, 1]
    betas[asset] = cov_asset / market_var_daily
print("Betas de cada activo respecto a la cartera de mercado:", betas)

# --- Sección f): Optimización sin ventas cortas ---
from scipy.optimize import minimize

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T.dot(cov_matrix).dot(weights))

n = len(returns_df.columns)
bounds = [(0, 1)] * n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

# f.1) Cartera de mínima varianza sin ventas cortas
def min_vol_objective(weights):
    return portfolio_volatility(weights, cov_annual)
    
res_min_ns = minimize(min_vol_objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
w_min_ns = res_min_ns.x
port_return_min_ns = sum(w_min_ns[i] * expected_returns_annual[asset] for i, asset in enumerate(returns_df.columns))
port_vol_min_ns = portfolio_volatility(w_min_ns, cov_annual)
print("Cartera de mínima varianza (sin ventas cortas):")
print("Pesos:", w_min_ns)
print("Retorno anual esperado:", port_return_min_ns)
print("Volatilidad anual:", port_vol_min_ns)

# f.2) Cartera tangente sin ventas cortas (maximizar el ratio de Sharpe)
def neg_sharpe(weights):
    port_return = sum(weights[i] * expected_returns_annual[asset] for i, asset in enumerate(returns_df.columns))
    port_vol = portfolio_volatility(weights, cov_annual)
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol != 0 else -1e6
    return -sharpe

res_tan_ns = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=constraints)
w_tan_ns = res_tan_ns.x
tan_return_ns = sum(w_tan_ns[i] * expected_returns_annual[asset] for i, asset in enumerate(returns_df.columns))
tan_vol_ns = portfolio_volatility(w_tan_ns, cov_annual)
print("Cartera tangente (sin ventas cortas):")
print("Pesos:", w_tan_ns)
print("Retorno anual esperado:", tan_return_ns)
print("Volatilidad anual:", tan_vol_ns)

# --- Sección g): Construcción de 3 carteras en la frontera sin ventas cortas ---
def optimize_portfolio_target(target_return):
    cons = constraints + [{'type': 'eq', 'fun': lambda w: sum(w[i] * expected_returns_annual[asset] for i, asset in enumerate(returns_df.columns)) - target_return}]
    res = minimize(min_vol_objective, np.ones(n)/n, bounds=bounds, constraints=cons)
    return res.x

# Definir 3 objetivos de retorno (bajo, medio y alto)
target_low = port_return_min_ns + 0.0005   # ligeramente superior a la mínima varianza
target_med = (port_return_min_ns + tan_return_ns) / 2
target_high = tan_return_ns - 0.0005          # ligeramente inferior al máximo
w_low = optimize_portfolio_target(target_low)
w_med = optimize_portfolio_target(target_med)
w_high = optimize_portfolio_target(target_high)

# Función para evaluar el desempeño en el último año
def performance_last_year(weights):
    last_year_returns = []
    for asset, df in datos.items():
        if df.empty or 'Date' not in df.columns or 'Price' not in df.columns:
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        cutoff = df['Date'].max() - pd.DateOffset(years=1)
        df_ly = df[df['Date'] > cutoff].copy()
        df_ly['log_return'] = np.log(df_ly['Price'] / df_ly['Price'].shift(1))
        df_ly = df_ly.dropna(subset=['log_return'])
        last_year_returns.append(df_ly.set_index('Date')['log_return'].rename(asset))
    if not last_year_returns:
        return None, None
    ly_df = pd.concat(last_year_returns, axis=1).dropna()
    port_daily = ly_df.dot(weights)
    cum_return = np.exp(port_daily.sum()) - 1
    return port_daily.mean(), cum_return

print("Cartera A (alto riesgo, sin ventas cortas):")
print("Pesos:", w_high)
mean_A, cum_A = performance_last_year(w_high)
print("Retorno diario promedio:", mean_A, "Retorno acumulado:", cum_A)

print("Cartera B (riesgo mediano, sin ventas cortas):")
print("Pesos:", w_med)
mean_B, cum_B = performance_last_year(w_med)
print("Retorno diario promedio:", mean_B, "Retorno acumulado:", cum_B)

print("Cartera C (bajo riesgo, sin ventas cortas):")
print("Pesos:", w_low)
mean_C, cum_C = performance_last_year(w_low)
print("Retorno diario promedio:", mean_C, "Retorno acumulado:", cum_C)

# --- Sección h): Modelo Black-Litterman (Bonus) ---
# Implementación simplificada. Se parte de los retornos implícitos del mercado (pi)
from scipy.linalg import inv

pi = np.array([expected_returns_annual[asset] for asset in returns_df.columns])
# Vista: el primer activo se espera que rinda 0.05 más que lo implícito
P = np.zeros((1, n))
P[0, 0] = 1
Q = np.array([pi[0] + 0.05])
omega = np.array([[0.1]])
tau = 0.025

# Fórmula Black-Litterman para obtener los retornos a posteriori
middle = inv(inv(tau * cov_annual) + P.T.dot(inv(omega)).dot(P))
mu_bl = middle.dot(inv(tau * cov_annual).dot(pi) + P.T.dot(inv(omega)).dot(Q))
    
def neg_sharpe_bl(weights):
    port_return = sum(weights[i] * mu_bl[i] for i in range(n))
    port_vol = portfolio_volatility(weights, cov_annual)
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol != 0 else -1e6
    return -sharpe

res_bl = minimize(neg_sharpe_bl, np.ones(n)/n, bounds=bounds, constraints=constraints)
w_bl = res_bl.x
bl_return = sum(w_bl[i] * mu_bl[i] for i in range(n))
bl_vol = portfolio_volatility(w_bl, cov_annual)
print("Cartera tangente Black-Litterman:")
print("Pesos:", w_bl)
print("Retorno anual esperado:", bl_return)
print("Volatilidad anual:", bl_vol)

mean_bl, cum_bl = performance_last_year(w_bl)
mean_tan, cum_tan = performance_last_year(w_tan_ns)
print("Desempeño último año - Black-Litterman: Retorno diario promedio:", mean_bl, "Retorno acumulado:", cum_bl)
print("Desempeño último año - Cartera tangente sin ventas cortas: Retorno diario promedio:", mean_tan, "Retorno acumulado:", cum_tan)
