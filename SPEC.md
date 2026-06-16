# BestBetWC - Sistema Inteligente para Optimizar Pronósticos en Polla Mundialista 2026

## 1. Análisis Crítico del Problema

### 1.1 El problema NO es predecir fútbol

El objetivo de este sistema **no es maximizar la precisión predictiva** sino **maximizar la puntuación esperada (Expected Score)** en una polla con reglas específicas.

Una estrategia óptima para esta polla puede ser **completamente distinta** de la estrategia óptima para predecir resultados. Ejemplo: si el 80% de los participantes predice Brasil 2-0, y tu modelo dice que hay un 35% de probabilidad de ese resultado, el valor esperado de elegir Brasil 2-0 es bajo porque:
- Si aciertas: ganas lo mismo que todos (bajo valor relativo)
- Si eliges un resultado alternativo con probabilidad razonable y aciertas: ganas el bono de predicción única (+2 pts)

### 1.2 Reglas exactas de la polla

| Concepto | Puntos |
|---|---|
| Acertar resultado (ganador/empate) | 2 pts |
| Acertar marcador exacto | 5 pts (reemplaza los 2 pts por resultado) |
| Acertar goles del local | 1 pt (adicional) |
| Acertar goles del visitante | 1 pt (adicional) |
| Bono 16avos (todos los clasificados) | 10 pts |
| Bono 8vos (todos los clasificados) | 8 pts |
| Bono 4tos (todos los clasificados) | 4 pts |
| Bono Semis (todos los clasificados) | 2 pts |
| Bono Final (todos los clasificados) | 5 pts |
| Predicción única (único con exacto) | 2 pts extra |

### 1.3 Ejemplos de puntuación por partido

**Caso 1:** Predices 2-1, sale 2-1
- Exacto: 5 pts
- Goles local (2=2): 1 pt
- Goles visitante (1=1): 1 pt
- **Total: 7 pts**

**Caso 2:** Predices 2-1, sale 2-0
- Resultado acertado: 2 pts
- Goles local (2=2): 1 pt
- Goles visitante (1≠0): 0 pts
- **Total: 3 pts**

**Caso 3:** Predices 2-1, sale 1-0
- Resultado acertado: 2 pts
- Goles local (2≠1): 0 pts
- Goles visitante (1≠0): 0 pts
- **Total: 2 pts**

**Caso 4:** Predices 2-1, sale 0-2
- Resultado desacertado: 0 pts
- Goles local (2≠0): 0 pts
- Goles visitante (1≠2): 0 pts
- **Total: 0 pts**

**Caso 5:** Predices 1-1, sale 1-1 (empate exacto)
- Exacto: 5 pts
- Goles local (1=1): 1 pt
- Goles visitante (1=1): 1 pt
- **Total: 7 pts**

### 1.4 Parámetros de la polla

| Parámetro | Valor |
|---|---|
| Torneo | Copa del Mundo FIFA 2026 |
| Participantes | 15 |
| Modificación pronósticos | Sí, hasta inicio del partido |
| Frecuencia | Por partido |
| Plataforma | pollamundial.org |
| Multiplicadores/comodines | No |
| Historial participantes | Pendiente (evaluar web scraping) |
| Desempate | Sin información |

### 1.5 Implicaciones estratégicas de las reglas

1. **El marcador exacto vale 5 pts vs 2 pts por resultado**: el ratio es 2.5x. Esto justifica invertir esfuerzo en predecir marcadores, no solo ganadores.

2. **Los goles acertados son adicionales**: acertar solo la cantidad de goles de un equipo (aunque falles el otro) vale 1 pt. Esto incentiva marcadores conservadores donde al menos un equipo tenga goles probables.

3. **El bono de predicción única (+2 pts)**: con 15 participantes, si eliges un marcador que solo tú predices y aciertas, ganas 7 pts (5 exacto + 2 única) + hasta 2 pts por goles = **9 pts máximo**. Esto incentiva la diferenciación estratégica.

4. **Bonos por ronda (bracket completo)**: son binarios (todo o nada) y requieren acertar TODOS los clasificados. El bono de 16avos (10 pts) es el más valioso. Esto incentiva predicciones conservadoras en fase de grupos para maximizar probabilidad de bracket correcto.

5. **Pronósticos modificables hasta inicio**: permite actualizar con información de última hora (alineaciones, clima, lesiones). El sistema debe soportar actualizaciones en tiempo real.

---

## 2. Fuentes de Datos y APIs

### 2.1 APIs de cuotas de apuestas (probabilidades implícitas del mercado)

| Fuente | Aporte | Prioridad | Costo |
|---|---|---|---|
| **The Odds API** | Cuotas de 50+ casas en tiempo real, movimiento de líneas, mercados de correct score, BTTS, over/under, asian handicap | ALTA | Free tier: 500 req/mes |
| **Pinnacle** | Líneas de cierre (closing odds), consideradas las más eficientes del mercado | ALTA | Via The Odds API |
| **Betfair Exchange** | Probabilidades del mercado de intercambio, volumen de apuestas | MEDIA | API gratuita con límites |

### 2.2 APIs de datos deportivos (features de rendimiento)

| Fuente | Aporte | Prioridad | Costo |
|---|---|---|---|
| **API-Football (RapidAPI)** | Resultados, alineaciones, estadísticas de partidos, lesiones, suspensiones | ALTA | Free: 100 req/día |
| **Football-Data.org** | Resultados históricos, tablas, fixtures | ALTA | Free tier disponible |
| **FBref** | xG, xGA, estadísticas avanzadas, datos a nivel jugador | ALTA | Web scraping |
| **Understat** | xG por partido, xG por jugador, mapas de tiros | MEDIA | Web scraping |
| **StatsBomb** | Datos de eventos detallados (pases, tiros, presiones) | BAJA (MVP) | Free para datos históricos |

### 2.3 APIs de contexto

| Fuente | Aporte | Prioridad |
|---|---|---|
| **Transfermarkt** | Valor de mercado, lesiones, suspensiones | MEDIA |
| **FIFA.com** | Rankings FIFA, calendario oficial | ALTA |
| **OpenWeatherMap** | Condiciones climáticas por sede | BAJA |

### 2.4 Web scraping - pollamundial.org

Evaluar viabilidad de extraer:
- Pronósticos de los 15 participantes
- Historial de aciertos por participante
- Clasificación en tiempo real
- Reglas completas de la polla

**Riesgos**: cambios en la estructura HTML, rate limiting, términos de servicio.

---

## 3. Features por Partido

### 3.1 Mercado de apuestas

```
odds_home_prob          # Probabilidad implícita local (normalizada, sin margen)
odds_draw_prob          # Probabilidad implícita empate
odds_away_prob          # Probabilidad implícita visitante
btts_yes_prob           # Probabilidad BTTS Sí
btts_no_prob            # Probabilidad BTTS No
over_05_prob            # Probabilidad Over 0.5
over_15_prob            # Probabilidad Over 1.5
over_25_prob            # Probabilidad Over 2.5
over_35_prob            # Probabilidad Over 3.5
over_45_prob            # Probabilidad Over 4.5
correct_score_probs     # Distribución de probabilidad de marcador exacto
asian_handicap_line     # Línea de hándicap asiático
asian_handicap_odds     # Cuotas del hándicap
odds_movement           # Delta entre apertura y cierre
closing_odds_home       # Cuota de cierre local
closing_odds_draw       # Cuota de cierre empate
closing_odds_away       # Cuota de cierre visitante
bookmaker_disagreement  # Desviación estándar entre casas
num_bookmakers          # Número de casas que ofrecen cuotas
```

### 3.2 Rendimiento deportivo

```
elo_home                # Elo rating local
elo_away                # Elo rating visitante
elo_diff                # Diferencia de Elo
fifa_rank_home          # Ranking FIFA local
fifa_rank_away          # Ranking FIFA visitante
fifa_rank_diff          # Diferencia de ranking FIFA
form_5_home             # Puntos en últimos 5 partidos (local)
form_5_away             # Puntos en últimos 5 partidos (visitante)
form_10_home            # Puntos en últimos 10 partidos (local)
form_10_away            # Puntos en últimos 10 partidos (visitante)
goals_scored_5_home    # Goles anotados últimos 5 (local)
goals_scored_5_away    # Goles anotados últimos 5 (visitante)
goals_conceded_5_home  # Goles recibidos últimos 5 (local)
goals_conceded_5_away  # Goles recibidos últimos 5 (visitante)
xg_home                 # xG promedio local
xg_away                 # xG promedio visitante
xga_home                # xGA promedio local
xga_away                # xGA promedio visitante
xg_diff                 # Diferencia de xG
home_performance        # Rendimiento como local
away_performance        # Rendimiento como visitante
h2h_wins_home           # Victorias H2H local
h2h_wins_away           # Victorias H2H visitante
h2h_draws               # Empates H2H
h2h_goals_avg           # Goles promedio H2H
possession_diff         # Diferencia de posesión promedio
shots_diff              # Diferencia de tiros promedio
shots_on_target_diff    # Diferencia de tiros al arco promedio
```

### 3.3 Contexto

```
injuries_home           # Número de lesionados clave (local)
injuries_away           # Número de lesionados clave (visitante)
suspensions_home        # Suspensiones (local)
suspensions_away        # Suspensiones (visitante)
rotation_risk_home      # Riesgo de rotación (local)
rotation_risk_away      # Riesgo de rotación (visitante)
match_importance        # Importancia del partido (0-1)
already_qualified_home  # Ya clasificado (local)
already_qualified_away  # Ya clasificado (visitante)
must_win_home           # Necesita ganar (local)
must_win_away           # Necesita ganar (visitante)
fatigue_home            # Fatiga (días desde último partido) (local)
fatigue_away            # Fatiga (días desde último partido) (visitante)
rest_days_home          # Días de descanso (local)
rest_days_away          # Días de descanso (visitante)
travel_distance_home    # Distancia de viaje (local)
travel_distance_away    # Distancia de viaje (visitante)
weather_condition       # Condición climática
temperature             # Temperatura
altitude                # Altitud de la sede
is_knockout             # Es eliminatoria (boolean)
round                   # Ronda (grupo, 16avos, 8vos, 4tos, semi, final)
```

### 3.4 Features de la polla (teoría de juegos)

```
ownership_estimate      # % estimado de participantes que elegirán este resultado
contrarian_value        # Valor de elegir contra el consenso
leverage_score          # Cuánto te diferencia esta apuesta del resto
popular_score_home      # Popularidad estimada del resultado local
popular_score_draw      # Popularidad estimada del empate
popular_score_away      # Popularidad estimada del resultado visitante
```

---

## 4. Modelo de Predicción

### 4.1 Distribución de goles y marcadores

El sistema debe estimar la **probabilidad de cada marcador exacto** (0-0, 1-0, 0-1, 1-1, 2-0, ..., hasta 5-5 o más).

#### Modelo base: Dixon-Coles

**Ventajas:**
- Corrige el sesgo del Poisson independiente para marcadores bajos (0-0, 1-0, 0-1, 1-1)
- Incluye factor de correlación entre goles de ambos equipos
- Parámetro de "baja inflación" (rho) que ajusta la probabilidad de marcadores con pocos goles
- Bien establecido en la literatura de predicción de fútbol
- Interpretable y calibrable

**Desventajas:**
- Asume distribución paramétrica fija
- No captura efectos no lineales complejos
- Requiere suficientes datos históricos para estimar parámetros

#### Modelo complementario: Bivariate Poisson

**Ventajas:**
- Captura correlación directa entre goles de ambos equipos
- Más flexible que Dixon-Coles para ciertos patrones
- Útil para modelar partidos donde el rendimiento de un equipo afecta al otro

**Desventajas:**
- Más parámetros que estimar
- Puede sobreajustar con datos limitados

#### Modelo ML: Gradient Boosting (XGBoost/LightGBM)

**Ventajas:**
- Captura interacciones no lineales entre features
- Maneja bien features heterogéneas (odds + stats + contexto)
- Robusto a outliers
- Feature importance interpretable

**Desventajas:**
- Requiere más datos de entrenamiento
- Menos calibrado probabilísticamente que modelos bayesianos
- Necesita post-procesamiento para obtener distribuciones de probabilidad

#### Modelo Bayesiano

**Ventajas:**
- Cuantifica incertidumbre de forma natural
- Permite incorporar priors (ej: rankings FIFA como prior de fuerza)
- Se actualiza con nuevos datos de forma natural
- Buena calibración probabilística

**Desventajas:**
- Computacionalmente más costoso
- Requiere especificar priors adecuados
- Más complejo de implementar

### 4.2 Estrategia de ensamble

```
Fase 1 (MVP): Dixon-Coles + Odds del mercado como prior
Fase 2: Dixon-Coles + Gradient Boosting (ensemble ponderado)
Fase 3: Ensemble bayesiano con todos los modelos
```

### 4.3 Output del modelo de predicción

Para cada partido, el sistema produce:

```
P(home_goals = i, away_goals = j)  para todo i,j en [0, 7]

De donde se derivan:
P(home_win)  = sum de P(i,j) donde i > j
P(draw)      = sum de P(i,j) donde i = j
P(away_win)  = sum de P(i,j) donde i < j

P(home_goals = k)  para todo k  (marginal de goles local)
P(away_goals = k)  para todo k  (marginal de goles visitante)
```

---

## 5. Capa de Optimización de Puntuación

### 5.1 Expected Score (EP) por marcador candidato

Para cada marcador candidato (i, j), calcular:

```
EP(i, j) = P(exacto) × 5
         + P(acertar_goles_local) × 1
         + P(acertar_goles_visitante) × 1
         + P(acertar_resultado) × 2  (solo si no es exacto)
         + P(predicción_única) × 2   (solo si es exacto)
```

Donde:

```
P(exacto) = P(home_goals = i, away_goals = j)

P(acertar_goles_local) = P(home_goals = i)
  (probabilidad marginal de que el local anote exactamente i goles)

P(acertar_goles_visitante) = P(away_goals = j)
  (probabilidad marginal de que el visitante anote exactamente j goles)

P(acertar_resultado) =
  si i > j: P(home_win) - P(exacto)
  si i = j: P(draw) - P(exacto)
  si i < j: P(away_win) - P(exacto)
  (restamos P(exacto) porque el exacto reemplaza al resultado)

P(predicción_única) = P(exacto) × P(nadie_más_elige_i_j)
  donde P(nadie_más_elige_i_j) = (1 - ownership(i,j))^14
  (14 = los otros 14 participantes)
```

### 5.2 Expected Score total del partido

```
EP_total(i, j) = P(exacto) × 5
               + P(home_goals = i) × 1
               + P(away_goals = j) × 1
               + P(resultado_sin_exacto) × 2
               + P(exacto) × (1 - ownership(i,j))^14 × 2
```

### 5.3 Selección óptima

```
pronóstico_óptimo = argmax_{(i,j)} EP_total(i, j)
```

**Importante:** El resultado más probable NO es necesariamente el mejor pronóstico. Un marcador menos probable pero poco popular puede tener mayor EP gracias al bono de predicción única.

### 5.4 Métricas de decisión

```
Expected Points (EP):           puntos esperados por partido
Expected Relative Gain (ERG):   puntos esperados vs participante promedio
Expected Rank:                  posición esperada después de la fecha
Win Probability:                probabilidad de terminar primero
Leverage Score:                 cuánto te diferencia esta apuesta
Ownership Estimate:             % estimado que elegirá el mismo resultado
Contrarian Value:               valor de ir contra el consenso
Upside Potential:               potencial de remontada
Risk of Ruin:                   probabilidad de perder posiciones
Calibration:                    calidad de las probabilidades estimadas
Closing Line Value (CLV):       comparación vs consenso final del mercado
```

---

## 6. Teoría de Juegos

### 6.1 Perfilado de jugadores

Para cada uno de los 15 participantes, estimar:

```
player_profile = {
    conservative_score:      0-1  (tiende a elegir favoritos)
    aggressive_score:        0-1  (tiende a elegir upsets)
    market_follower:         0-1  (sigue las cuotas de apuestas)
    favorite_bias:           0-1  (sobrevalora equipos populares)
    recency_bias:            0-1  (sobrepondera resultados recientes)
    home_bias:               0-1  (favorece al local)
    round_of_16_accuracy:    0-1  (histórico en 16avos)
    quarterfinal_accuracy:   0-1  (histórico en cuartos)
    exact_score_frequency:   0-1  (frecuencia de elegir marcadores arriesgados)
    popular_teams_bias:      {}   (equipos que sobrevalora)
    unpopular_teams_bias:    {}   (equipos que infravalora)
}
```

### 6.2 Modelado de decisiones

Inferir para cada participante:
- Distribución de marcadores elegidos por tipo de partido
- Sesgos sistemáticos (ej: siempre predice victorias de equipos sudamericanos)
- Correlación con cuotas de mercado
- Tasa de acierto histórica por tipo de resultado

### 6.3 Estrategia adaptativa

```
SI posición_actual == 1 (liderando):
    estrategia = MINIMIZAR_RIESGO
    - Elegir marcadores de alta probabilidad
    - Evitar diferenciación innecesaria
    - Maximizar EP mínimo garantizado

SI posición_actual en [2, 5]:
    estrategia = EQUILIBRADA
    - Mezclar predicciones seguras con algunas diferenciadas
    - Buscar EP máximo con riesgo controlado

SI posición_actual en [6, 10]:
    estrategia = DIFERENCIACIÓN
    - Priorizar predicciones con alto Contrarian Value
    - Buscar bonos de predicción única
    - Aceptar mayor varianza

SI posición_actual en [11, 15]:
    estrategia = ALTO_RIESGO
    - Buscar marcadores de alto EP con baja ownership
    - Maximizar Upside Potential
    - Aceptar Risk of Ruin alto
```

### 6.4 Game Theory Layer

Evaluar rigurosamente cuándo aportan valor:

| Concepto | Aplicabilidad | Justificación |
|---|---|---|
| **Nash Equilibrium** | MEDIA | Útil cuando hay información de los pronósticos de otros. En la práctica, no todos los jugadores son racionales. |
| **Best Response** | ALTA | Dado el perfil estimado de los otros 14, calcular la mejor respuesta. Directamente aplicable. |
| **Regret Minimization** | MEDIA | Útil para ajustar la estrategia a lo largo del torneo. Minimizar el regret acumulado vs la mejor estrategia retrospectiva. |
| **Multi-Armed Bandits** | BAJA | Podría usarse para explorar/explotar tipos de predicción, pero con solo 64 partidos es limitado. |
| **Bayesian Games** | ALTA | Modelar la incertidumbre sobre las estrategias de los otros jugadores como tipos bayesianos. |
| **Opponent Modeling** | ALTA | Con historial de pronósticos, construir modelos predictivos de lo que elegirán los demás. |

---

## 7. Simulación Monte Carlo

### 7.1 Simulador de torneos

```
Para cada simulación (N = 100,000):
    1. Simular fase de grupos usando distribución de goles por partido
    2. Determinar clasificados por grupo
    3. Simular eliminatorias (16avos → final)
    4. Calcular puntuación de cada estrategia candidata
    5. Registrar posición final

Output:
    - Distribución de posiciones finales para cada estrategia
    - Probabilidad de terminar en cada posición (1-15)
    - Expected Rank por estrategia
    - Win Probability por estrategia
    - Risk of Ruin por estrategia
```

### 7.2 Simulador de participantes

```
Para cada participante simulado:
    1. Generar perfil basado en datos históricos (o priors si no hay datos)
    2. Simular pronósticos basados en su perfil
    3. Calcular puntuación esperada del participante
    4. Comparar con la estrategia del usuario
```

---

## 8. Ciencia de Datos

### 8.1 Esquema de base de datos

```sql
-- Equipos
teams (id, name, fifa_rank, elo_rating, confederation, group_id)

-- Partidos
matches (id, home_team_id, away_team_id, datetime, venue, round, 
         altitude, weather, status)

-- Cuotas de apuestas
odds (id, match_id, bookmaker, timestamp, home_odds, draw_odds, 
      away_odds, over_odds, under_odds, btts_yes, btts_no)

-- Mercado de marcador exacto
correct_score_odds (id, match_id, bookmaker, timestamp, 
                    home_goals, away_goals, odds)

-- Features de rendimiento
team_form (id, team_id, match_id, goals_scored, goals_conceded,
           xg, xga, possession, shots, shots_on_target, 
           result, is_home)

-- H2H
head_to_head (id, team_a_id, team_b_id, match_date, 
              goals_a, goals_b, competition)

-- Lesiones y contexto
injuries (id, team_id, player_name, injury_type, expected_return)

-- Pronósticos del sistema
system_predictions (id, match_id, timestamp, home_goals, away_goals,
                    ep_score, ownership_estimate, contrarian_value,
                    confidence, strategy_mode)

-- Pronósticos de participantes
participant_predictions (id, match_id, participant_id, 
                         home_goals, away_goals, timestamp)

-- Resultados
results (id, match_id, home_goals, away_goals, 
         home_penalties, away_penalties)

-- Puntuaciones
scores (id, match_id, participant_id, result_pts, exact_pts,
        goals_home_pts, goals_away_pts, unique_pts, 
        round_bonus_pts, total_pts)

-- Clasificación
standings (id, participant_id, round, total_points, position)

-- Perfiles de participantes
participant_profiles (id, participant_id, conservative_score,
                      aggressive_score, market_follower, 
                      favorite_bias, recency_bias, home_bias,
                      updated_at)
```

### 8.2 Pipeline ETL

```
1. EXTRACCIÓN
   ├── API-Football → partidos, resultados, alineaciones
   ├── The Odds API → cuotas en tiempo real
   ├── FBref/Understat → xG, stats avanzadas (scraping)
   ├── FIFA.com → rankings, calendario
   └── pollamundial.org → pronósticos participantes (scraping)

2. TRANSFORMACIÓN
   ├── Normalización de cuotas → probabilidades implícitas (sin margen)
   ├── Cálculo de Elo ratings actualizados
   ├── Cálculo de forma reciente (5, 10 partidos)
   ├── Cálculo de xG/xGA acumulados
   ├── Feature engineering (diferencias, ratios, tendencias)
   └── Matching temporal (features disponibles antes del partido)

3. CARGA
   ├── Base de datos principal (PostgreSQL/SQLite)
   ├── Cache de cuotas actualizadas
   └── Snapshot de features por partido
```

### 8.3 Validación temporal

**NO usar cross-validation aleatorio.** Usar validación temporal:

```
Entrenar con datos hasta tiempo T
Validar con datos de T a T+1
Avanzar T y repetir

Métricas en cada fold temporal:
- Log Loss
- Brier Score
- Calibration Error
- Ranked Probability Score (RPS)
- Expected Score obtenido (la métrica principal)
```

### 8.4 Métricas de evaluación

| Métrica | Objetivo | Target |
|---|---|---|
| **Expected Score** | Puntos esperados por partido | Maximizar |
| **Log Loss** | Calidad probabilística | < 0.9 |
| **Brier Score** | Error cuadrático de probabilidades | < 0.2 |
| **Calibration Error** | Calibración de probabilidades | < 0.05 |
| **RPS** | Ranked Probability Score | < 0.2 |
| **CLV** | Closing Line Value | > 0 (positivo) |
| **Exact Score Hit Rate** | % de marcadores exactos acertados | > 12% |
| **Result Hit Rate** | % de resultados acertados | > 50% |

### 8.5 Backtesting

```
1. Tomar Mundiales anteriores (2014, 2018, 2022)
2. Simular la polla con las reglas actuales
3. Comparar estrategias:
   a. Baseline: siempre elegir el favorito con marcador 1-0
   b. Market: seguir cuotas de apuestas
   c. Model: usar predicciones del modelo
   d. Optimized: usar EP con optimización
   e. Contrarian: usar EP + diferenciación
4. Reportar Expected Score, posición final, Win Probability
```

---

## 9. Arquitectura

### 9.1 Stack tecnológico

| Componente | Tecnología | Justificación |
|---|---|---|
| Lenguaje principal | **Python 3.11+** | Ecosistema de ML, APIs, scraping |
| Base de datos | **SQLite** (MVP) → **PostgreSQL** (producción) | Simple para MVP, escalable después |
| ML Framework | **scikit-learn + XGBoost + PyMC** | Estándar, bien documentado |
| Optimización | **scipy.optimize** | Suficiente para EP maximization |
| Simulación | **numpy** | Monte Carlo eficiente |
| APIs | **httpx + asyncio** | Async para múltiples APIs |
| Scraping | **playwright** | Para pollamundial.org y FBref |
| CLI | **typer** | Interfaz de línea de comandos |
| Visualización | **matplotlib + plotly** | Análisis y reportes |
| Testing | **pytest** | Estándar de la industria |
| Tareas | **celery + redis** (futuro) | Scheduling de ETL |

### 9.2 Estructura del proyecto

```
BestBetWC/
├── SPEC.md                    # Este documento
├── AGENTS.md                  # Instrucciones para agentes de IA
├── pyproject.toml             # Configuración del proyecto
├── .env.example               # Variables de entorno (API keys)
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuración global
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py          # Modelos SQLAlchemy
│   │   ├── connection.py      # Conexión a BD
│   │   └── migrations/        # Migraciones
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── odds_api.py        # The Odds API
│   │   ├── api_football.py    # API-Football
│   │   ├── fbref.py           # FBref scraper
│   │   ├── fifa.py            # FIFA rankings
│   │   └── polla_scraper.py   # pollamundial.org scraper
│   ├── features/
│   │   ├── __init__.py
│   │   ├── market.py          # Features de mercado
│   │   ├── performance.py     # Features de rendimiento
│   │   ├── context.py         # Features de contexto
│   │   └── pipeline.py        # Pipeline de features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dixon_coles.py     # Modelo Dixon-Coles
│   │   ├── poisson.py         # Poisson base
│   │   ├── gradient_boost.py  # XGBoost/LightGBM
│   │   ├── bayesian.py        # Modelo bayesiano
│   │   └── ensemble.py        # Ensemble de modelos
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── expected_score.py  # Cálculo de EP
│   │   ├── strategy.py        # Selector de estrategia
│   │   └── bracket.py         # Optimización de bracket
│   ├── game_theory/
│   │   ├── __init__.py
│   │   ├── profiling.py       # Perfilado de jugadores
│   │   ├── opponent_model.py  # Modelado de oponentes
│   │   ├── best_response.py   # Best response calculator
│   │   └── adaptive.py        # Estrategia adaptativa
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── tournament.py      # Simulador de torneos
│   │   ├── participants.py    # Simulador de participantes
│   │   └── monte_carlo.py     # Motor Monte Carlo
│   └── cli/
│       ├── __init__.py
│       ├── main.py            # CLI principal
│       ├── predict.py         # Comando de predicción
│       ├── simulate.py        # Comando de simulación
│       └── report.py          # Comando de reportes
├── tests/
│   ├── test_expected_score.py
│   ├── test_dixon_coles.py
│   ├── test_optimization.py
│   ├── test_simulation.py
│   └── test_game_theory.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_model_calibration.ipynb
│   └── 03_strategy_analysis.ipynb
└── data/
    ├── raw/                   # Datos crudos
    ├── processed/             # Datos procesados
    └── predictions/           # Predicciones generadas
```

### 9.3 Evolución por fases

```
FASE 1 - MVP (Semanas 1-2)
├── Dixon-Coles con datos de API-Football
├── Cálculo de Expected Score
├── Recomendaciones basadas en EP
├── CLI para generar pronósticos
└── Backtesting con Mundiales 2018, 2022

FASE 2 - Modelos Estadísticos (Semanas 3-4)
├── Integración de The Odds API
├── Features de mercado completas
├── Ensemble Dixon-Coles + mercado
├── Validación temporal
└── Dashboard de métricas

FASE 3 - Machine Learning (Semanas 5-6)
├── XGBoost con todas las features
├── Ensemble ponderado
├── Feature importance analysis
├── Hyperparameter tuning
└── Mejora de calibración

FASE 4 - Teoría de Juegos (Semanas 7-8)
├── Web scraping de pollamundial.org
├── Perfilado de participantes
├── Ownership estimation
├── Contrarian value optimization
└── Estrategia adaptativa

FASE 5 - Sistema Adaptativo (Semanas 9-10)
├── Simulación Monte Carlo completa
├── Best response dinámico
├── Actualización en tiempo real
├── Alertas de valor
└── Reportes automáticos
```

---

## 10. Roadmap de Implementación

### MVP (Fase 1) - Entregables inmediatos

1. **Motor Dixon-Coles** con datos históricos de selecciones
2. **Calculadora de Expected Score** con las reglas exactas de la polla
3. **Generador de pronósticos** que maximice EP por partido
4. **CLI** para consultar pronósticos óptimos antes de cada partido
5. **Backtesting básico** con Mundiales 2018 y 2022

### Post-MVP

6. Integración de cuotas en tiempo real
7. Features avanzadas (xG, forma, contexto)
8. Modelo ML ensemble
9. Web scraping de la polla
10. Teoría de juegos y diferenciación
11. Simulación Monte Carlo del torneo completo
12. Dashboard web

---

## 11. Riesgos y Limitaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Datos insuficientes de selecciones | ALTO | Usar datos de clubes como proxy, priors bayesianos |
| Overfitting en modelos ML | ALTO | Validación temporal estricta, regularización |
| APIs con rate limiting | MEDIO | Cache agresivo, múltiples fuentes |
| Web scraping de pollamundial.org bloqueado | MEDIO | Datos manuales como fallback |
| Cambios de reglas de la polla | BAJO | Configurar reglas como parámetros |
| Lesiones/suspensiones de última hora | ALTO | Actualización pre-partido, features de contexto |
| Modelo mal calibrado | ALTO | Monitoreo continuo de calibration error |
| Pocos participantes (15) limita teoría de juegos | MEDIO | Con 15 jugadores hay suficiente señal si hay historial |
| Mundial 2026 formato expandido (48 equipos) | MEDIO | Más partidos = más datos pero más incertidumbre en equipos nuevos |

---

## 12. Si el objetivo fuera ganar la polla (no solo predecir)

1. **No intentes predecir todos los partidos perfectamente.** Concentra tu edge en los partidos donde el mercado es menos eficiente y donde la diferenciación tiene más valor.

2. **El bono de predicción única (+2 pts) es tu arma secreta.** Con 15 participantes, los marcadores populares (1-0, 2-1, 1-1) serán elegidos por 5-8 personas. Si aciertas un marcador que nadie más eligió, ganas 2 pts extra. En un torneo de 64 partidos, esto puede ser la diferencia entre ganar y perder.

3. **Los bonos de bracket son binarios y de alto valor.** El bono de 16avos (10 pts) requiere acertar TODOS los clasificados. Esto vale más que 2 partidos perfectos. Prioriza predicciones conservadoras en fase de grupos para maximizar la probabilidad de bracket correcto.

4. **Adapta tu estrategia a tu posición en la tabla.** Si vas primero, juega conservador. Si vas atrás, busca diferenciación agresiva. El sistema debe calcular automáticamente la estrategia óptima según tu posición.

5. **El closing line value (CLV) es tu brújula.** Si tus predicciones consistently beat the closing line, tu modelo es bueno. Si no, el mercado es mejor que tu modelo y deberías seguir más al mercado.

6. **Los goles acertados son puntos "gratuitos".** Incluso si fallas el resultado, acertar los goles de un equipo te da 1 pt. En marcadores donde un equipo es muy favorito, predecir sus goles correctos (aunque falles el otro equipo) es una estrategia de bajo riesgo.

7. **La teoría de juegos importa más que mejorar el modelo de 70% a 75%.** Con 15 participantes, el ganador no es quien predice mejor sino quien toma mejores decisiones relativas. Si todos predicen Brasil 2-0 y tú predices Brasil 2-1, ambos aciertan el resultado pero tú además ganas puntos por goles distintos.

8. **Monitorea y actualiza.** Los pronósticos se pueden modificar hasta el inicio. Usa información de última hora (alineaciones, clima) para actualizar. El sistema debe generar alertas cuando hay valor en cambiar un pronóstico.
