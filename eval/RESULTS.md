# Результаты экспериментов

Дата: 2026-11-08

Корпус: 6 статей Википедии о дифференциальных уравнениях, 132 чанка.
Golden dataset: 30 вопросов, 60 явных gold-чанков (1–3 на вопрос).
Эмбеддер: `intfloat/multilingual-e5-small`, 512 токенов.
Векторный индекс: Qdrant, cosine, top_k = 5.

## Метрики

- **article_hit@k** — ожидаемая статья попала в топ-k;
- **section_hit@k** — ожидаемый раздел попал в топ-k;
- **recall@k** — какая доля gold-чанков попала в топ-k (главная метрика);
- **precision@k** — какая доля top-k является gold-чанками;
- **MRR** — средний обратный ранг первого правильного попадания.

Precision@5 структурно ограничен: gold-чанков 1–3, а слотов всегда 5.
Поэтому recall важнее.

## Retrieval: до и после

| Вариант | recall@5 | Δ recall | section_hit@3 | Δ section | article_hit@3 | MRR | Δ MRR |
|---|---|---|---|---|---|---|---|
| baseline | 37.2% | — | 50.0% | — | 83.3% | 0.787 | — |
| header | 45.6% | +8.4 п.п. | 63.3% | +13.3 п.п. | 80.0% | 0.765 | −0.022 |
| summary_en | 51.1% | +13.9 п.п. | 76.7% | +26.7 п.п. | 90.0% | 0.800 | +0.013 |
| summary_ru (long) | 63.3% | +26.1 п.п. | 80.0% | +30.0 п.п. | 96.7% | 0.936 | +0.149 |
| summary_short | 51.1% | +13.9 п.п. | 60.0% | +10.0 п.п. | 96.7% | **0.919** | +0.132 |
| chunk_context | 56.1% | +18.9 п.п. | 63.3% | +13.3 п.п. | 90.0% | 0.853 | +0.066 |
| **combined** | **65.6%** | **+28.4 п.п.** | **80.0%** | **+30.0 п.п.** | 93.3% | 0.894 | +0.107 |

`combined` = короткое саммари раздела + chunk-specific context.

### Precision@5 — диагностическая метрика

Precision@5 структурно ограничен: gold-чанков 1–3, а слотов всегда 5,
поэтому максимум равен 20–60%. Эти цифры не входят в основную таблицу
качества.

| Вариант | precision@5 |
|---|---|
| baseline | 14.7% |
| header | 16.7% |
| summary_en | 19.3% |
| summary_ru (long) | 24.0% |
| summary_short | 20.7% |
| chunk_context | 22.0% |
| combined | 26.0% |
## Токены и лимит e5 (512)

| Вариант | max токенов | > 512 | контекст обрезан |
|---|---|---|---|
| baseline | 507 | 0 | 0 |
| header (с бюджетом) | 510 | 0 | 1 |
| summary_en | 569 | 6 | — |
| summary_ru (long) | 590 | 7 | — |
| summary_short | 510 | 0 | 4 |
| chunk_context | 510 | 0 | 7 |
| combined | 510 | 0 | 9 |

Важно: `summary_en` и `summary_ru (long)` выходили за 512, поэтому e5 обрезал
хвост исходного чанка. Это confound: нельзя однозначно сказать, что прирост
дали только саммари. В ноутбуке 03 бюджет соблюдён: `over_512 = 0`, исходный
чанк не обрезается, при необходимости укорачивается только контекст.

## Recall@5 по темам

| Тема | baseline | summary_ru (long) | combined |
|---|---|---|---|
| integrating_factor | 0.000 | 0.833 | 0.667 |
| linear_ode | 0.633 | 0.533 | 0.700 |
| lipschitz | 0.208 | 0.292 | 0.417 |
| multi_hop | 0.433 | 0.533 | 0.700 |
| ode | 0.250 | 0.875 | 0.875 |
| picard | 0.000 | 0.500 | 0.300 |
| wronskian | 1.000 | 1.000 | 1.000 |

## Context compaction

`eval/notebooks/04_context_compaction.ipynb` — сжатие retrieved-контекста
(extractive, бюджет 500 токенов).

### Качество ответа

| Режим | correctness | faithfulness | citations | ответов со ссылками |
|---|---|---|---|---|
| full | 3.73 | 4.87 | 4.73 | 100% |
| compact | 3.50 | 4.73 | 4.63 | 100% |

### Стоимость

| Режим | context tokens | reduction |
|---|---|---|
| full | 1072.3 | — |
| compact | **497.7** | **52.1%** |

### Correctness по темам

| Тема | full | compact | Δ |
|---|---|---|---|
| integrating factor | 4.67 | 4.67 | 0.00 |
| linear ODE | 3.80 | 2.80 | −1.00 |
| lipschitz | 3.00 | 3.25 | +0.25 |
| multi-hop | 4.20 | 3.80 | −0.40 |
| ODE | 3.50 | 3.25 | −0.25 |
| Picard | 2.60 | 2.60 | 0.00 |
| Wronskian | 4.75 | 4.75 | 0.00 |

### Token reduction по темам

| Тема | reduction |
|---|---|
| integrating factor | 62% |
| linear ODE | 50% |
| lipschitz | 52% |
| multi-hop | 54% |
| ODE | 50% |
| Picard | 47% |
| Wronskian | 54% |

Просадка качества умеренная, но формулоёмкие разделы (`linearode-charpoly-07`)
страдают сильнее: correctness 5 → 2. Следующий шаг — sweep бюджета
500 / 700 / 900 и formula-aware splitting.

## RAG vs агент

`eval/notebooks/05_agent_vs_rag.ipynb` — 30 вопросов, одинаковый judge.
RAG видит top-5, агент — результаты своих вызовов инструментов.

### Качество ответа

| Режим | correctness | faithfulness | citations | ответов со ссылками |
|---|---|---|---|---|
| RAG | 3.73 | **4.90** | **4.77** | 100% |
| агент | **4.63** | 4.53 | 4.60 | 100% |

### Стоимость

| Режим | latency mean | latency p50 | iterations | tool calls |
|---|---|---|---|---|
| RAG | 3.1 с | 3.1 с | 1 | 0 |
| агент | 7.9 с | 8.1 с | 2.40 | 2.03 |

### Correctness по темам

| Тема | RAG | агент | Δ |
|---|---|---|---|
| integrating factor | 4.67 | 5.00 | +0.33 |
| linear ODE | 3.80 | 4.00 | +0.20 |
| lipschitz | 3.25 | 4.75 | +1.50 |
| multi-hop | 4.00 | 5.00 | +1.00 |
| ODE | 3.75 | 5.00 | +1.25 |
| Picard | 2.40 | 4.00 | +1.60 |
| Wronskian | 4.75 | 5.00 | +0.25 |

### Tool-choice

| Метрика | Значение |
|---|---|
| вопросов | 6 |
| tool_choice_accuracy | 100% |
| exact_tool_set_accuracy | 100% |
| tool_precision / recall / F1 | 1.00 / 1.00 / 1.00 |
| sequence_accuracy | 100% |
| unnecessary_tool_rate | 0% |

Все 6 вопросов решены правильным набором инструментов и в правильном порядке,
включая multi-tool `search_wikipedia` → `run_python`.

### Вывод

Агент лучше на сложных темах: Picard +1.6, Lipschitz +1.5, ODE +1.25,
multi-hop +1.0. По вопросам: 11 улучшений, 18 совпадений, 1 регрессия
(`linearode-charpoly-07`, 5 → 4).

Цена — latency в 2.6 раза выше и более низкая faithfulness: агент добавляет
детали, которые не подтверждаются пассажами. Для простых определений RAG
достаточно и он точнее.

Практичная схема — гибридная маршрутизация: простой вопрос → RAG,
multi-hop и вычисления → агент.

## RAGAS

`eval/notebooks/06_ragas.ipynb` — 30 вопросов, RAG-режим.
Судья — DeepSeek, эмбеддинги — e5-small. Все метрики от 0 до 1.

### Сводка

| Метрика | Значение |
|---|---|
| faithfulness | **0.893** |
| answer relevancy | 0.782 |
| context precision | 0.578 |
| context recall | 0.675 |
| answer correctness | 0.520 |

Ошибок — 0. RAGAS занял ~555 секунд, генерация ответов — ~87 секунд.

### По темам

| Тема | n | faithfulness | answer relevancy | context precision | context recall | answer correctness |
|---|---|---|---|---|---|---|
| integrating factor | 3 | 0.981 | 0.859 | 0.917 | 1.000 | **0.807** |
| linear ODE | 5 | 0.920 | 0.945 | 0.572 | 0.633 | 0.536 |
| lipschitz | 4 | 0.865 | 0.955 | 0.275 | 0.438 | 0.423 |
| multi-hop | 5 | 0.894 | 0.765 | 0.720 | 0.800 | 0.483 |
| ODE | 4 | 0.842 | 0.685 | 0.493 | 0.625 | 0.477 |
| Picard | 5 | 0.802 | 0.377 | 0.440 | 0.433 | 0.467 |
| Wronskian | 4 | 0.982 | 0.971 | 0.717 | 0.917 | 0.539 |

### Что видно

- **Faithfulness 0.893** — ответы почти не выдумывают. Проблема не в
  галлюцинациях, а в retrieval.
- **Context precision 0.578 / recall 0.675** — в top-5 много шума, а часть
  нужных фактов не находится. Хуже всего Lipschitz и Picard.
- **Answer relevancy 0.782**, но у пяти вопросов ровно 0.0: модель честно
  отвечает «в контексте этого нет» (`picard-theorem-14`,
  `picard-nonunique-16`, `multi-reduction-system-30` и др.).
- **Answer correctness 0.520** — RAGAS строже кастомного judge; лучший
  результат у integrating factor (0.807), худший — у lipschitz (0.423).

RAGAS подтверждает: узкое место — retrieval, а не генерация. Следующие шаги:
reranking, увеличение top-k, маршрутизация multi-hop вопросов в агент.

## Выводы

- Саммари раздела и chunk-specific context заметно улучшают retrieval.
- Лучший recall@5 — у `combined`: 37.2% → 65.6%.
- Лучший MRR — у `summary_short`: 0.919.
- `summary_ru (long)` даёт 63.3% recall, но с обрезкой исходного текста;
  результат не полностью чистый.
- Самые проблемные темы остаются Picard и ODE; Wronskian стабильно 100%.
- Сжатие контекста уменьшает его на 52% при умеренной просадке качества.
- Агент повышает correctness на сложных темах (Picard +1.6, Lipschitz +1.5,
  ODE +1.25), но в 2.6 раза медленнее RAG и менее точен по faithfulness.
- RAGAS: faithfulness 0.893 при context precision/recall 0.578/0.675 —
  ответы заземлены, но retrieval не находит всё необходимое.
- Следующий шаг — reranking/гибридный поиск, RAGAS для агента,
  затем гибридная маршрутизация RAG / агент.

## Файлы

- `eval/notebooks/01_retrieval_baseline.ipynb`
- `eval/notebooks/02_chunk_summary.ipynb`
- `eval/notebooks/03_chunk_context.ipynb`
- `eval/notebooks/04_context_compaction.ipynb`
- `eval/notebooks/05_agent_vs_rag.ipynb`
- `eval/notebooks/06_ragas.ipynb`
- `eval/results/retrieval_baseline.json`
- `eval/results/chunk_summary_experiment.json`
- `eval/results/chunk_context_experiment.json`
- `eval/results/context_compaction_experiment.json`
- `eval/results/agent_vs_rag_experiment.json`
- `eval/results/ragas_experiment.json`
