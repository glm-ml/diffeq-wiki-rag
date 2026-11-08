# Golden dataset и оценка

Здесь лежит эталонный набор вопросов, скрипты валидации и эксперименты.
Сводная таблица метрик: [`RESULTS.md`](RESULTS.md).

## Файлы

```
eval/
├── golden/
│   ├── qa.jsonl          # 30 вопросов
│   ├── gold_chunks.jsonl # эталонные чанки: 1–3 на вопрос
│   └── tool_qa.jsonl     # 6 вопросов для проверки выбора инструментов
├── notebooks/            # эксперименты
├── results/              # JSON с метриками
├── metrics.py            # hit / recall / precision / MRR
├── tool_metrics.py       # tool-choice precision / recall / F1
├── agent_eval.py         # прогон агента и разбор трейса
├── ragas_eval.py         # RAGAS: faithfulness, context precision/recall
├── summaries.py          # генерация саммари разделов
├── select_gold_chunks.py # подбор эталонных чанков
├── validate_golden.py    # проверка схемы и соответствия корпусу
└── README.md
```

## Эталонные чанки

`gold_chunks.jsonl` содержит по 1–3 чанка на вопрос:

```json
{
  "id": "picard-theorem-14",
  "gold_chunk_ids": ["Picard–Lindelöf theorem::Theorem::0"],
  "gold_evidence": ["Let D ⊂ R × R^n be a closed rectangle..."],
  "method": "manual"
}
```

`chunk_id` строится при индексации:

```
<article>::<section>[::<subsection>]::<chunk_index>
```

Например: `Wronskian::Definition::0`.

`method`:
- `auto` — выбран по семантической близости к reference_answer;
- `manual` — поправлен вручную после просмотра кандидатов.

Скрипт `select_gold_chunks.py` только предлагает кандидатов; итоговый файл
нужно просмотреть глазами.

## Схема записи

```json
{
  "id": "lipschitz-def-01",
  "topic": "lipschitz",
  "type": "definition",
  "question": "Что такое условие Липшица?",
  "expected_articles": ["Lipschitz continuity"],
  "expected_sections": ["Definitions"],
  "reference_answer": "Краткий эталонный ответ по учебнику...",
  "reference_source": "Coddington & Levinson, Ch. 1; Boyce & DiPrima, 9th ed., §2.8",
  "key_facts": ["K ≥ 0", "d_Y(...) ≤ K d_X(...)"],
  "must_cite": ["Lipschitz continuity"],
  "difficulty": "easy"
}
```

Поля:

- `topic` — тема: `lipschitz`, `linear_ode`, `ode`, `picard`, `integrating_factor`, `wronskian`, `multi_hop`;
- `type` — `definition`, `theorem`, `method`, `example`, `properties`, `multi_hop`;
- `expected_articles` — какие статьи должны попасть в топ-k;
- `expected_sections` — какие разделы считаются правильными;
- `reference_answer` — эталонный ответ (парафраз учебника, не цитата);
- `reference_source` — откуда взят эталон;
- `key_facts` — факты, которые должны быть в ответе;
- `must_cite` — источники, которые обязаны быть в цитатах;
- `difficulty` — `easy`, `medium`, `hard`.

## Откуда эталонные ответы

- **Boyce & DiPrima, Elementary Differential Equations and Boundary Value Problems, 9th ed.**
  - §2.1 — линейные уравнения и интегрирующий множитель;
  - §2.6 — точные уравнения и интегрирующие множители;
  - §2.8 — теорема существования и единственности;
  - §3.1–3.4 — уравнения второго порядка, вронскиан, понижение порядка;
  - §4.1–4.2 — уравнения n-го порядка;
  - Ch. 7 — системы первого порядка.
- **Coddington & Levinson, Theory of Ordinary Differential Equations, Ch. 1** —
  существование, единственность, условие Липшица, итерации Пикара.

Ответы — краткие парафразы, без копирования текста учебников.

## Запуск валидации

```powershell
uv run python -m eval.validate_golden
```

Скрипт проверяет:
- уникальность `id`;
- наличие обязательных полей;
- что `expected_articles` и `expected_sections` реально есть в корпусе;
- что `must_cite` входит в `expected_articles`;
- что у каждого вопроса есть 1–3 gold-чанка и все они существуют в корпусе;
- распределение по темам и сложности.

## Метрики

Retrieval:
- `article_hit@k` — ожидаемая статья попала в топ-k;
- `section_hit@k` — ожидаемый раздел попал в топ-k;
- `recall@k` — какая доля gold-чанков попала в топ-k;
- `precision@k` — какая доля top-k является gold-чанками;
- MRR.

Для RAG важнее **recall**: LLM нужны все необходимые факты. Precision тоже
важен, потому что мусор в контексте увеличивает стоимость и мешает ответу.

RAGAS-метрики (0–1):

- `faithfulness` — подтверждаются ли утверждения ответа контекстом;
- `answer_relevancy` — отвечает ли ответ на вопрос;
- `context_precision` — стоят ли релевантные чанки выше нерелевантных;
- `context_recall` — покрывает ли контекст эталонный ответ;
- `answer_correctness` — смысловая и фактическая близость к эталону.

RAGAS строже кастомного judge: это нормализованные 0–1, а не 1–5, и он
чувствителен к качеству retrieval.

## Эксперименты

`eval/notebooks/01_retrieval_baseline.ipynb` — baseline на 30 вопросах:

| Метрика | Значение |
|---|---|
| article hit@3 | 83.3% |
| section hit@3 | 50.0% |
| recall@5 | 37.2% |
| MRR | 0.787 |

`precision@5` — диагностическая метрика, см. `RESULTS.md`.

`eval/notebooks/02_chunk_summary.ipynb` — header vs section summary:

| Вариант | recall@5 | section hit@3 | article hit@3 | MRR |
|---|---|---|---|---|
| baseline | 37.2% | 50.0% | 83.3% | 0.787 |
| header | 45.6% | 63.3% | 80.0% | 0.765 |
| summary_en | 51.1% | 76.7% | 90.0% | 0.800 |
| **summary_ru** | **63.3%** | **80.0%** | **96.7%** | **0.936** |

Саммари разделов кэшируются в `eval/results/section_summaries.json`.

`eval/notebooks/03_chunk_context.ipynb` — короткое саммари и chunk-specific context:

| Вариант | recall@5 | section hit@3 | article hit@3 | MRR |
|---|---|---|---|---|
| baseline | 37.2% | 50.0% | 83.3% | 0.787 |
| header | 45.6% | 63.3% | 80.0% | 0.765 |
| summary_short | 51.1% | 60.0% | 96.7% | 0.919 |
| chunk_context | 56.1% | 63.3% | 90.0% | 0.853 |
| **combined** | **65.6%** | **80.0%** | 93.3% | 0.894 |

Все варианты укладываются в 512 токенов e5; контекст при необходимости
обрезается, исходный чанк не меняется.

`eval/notebooks/04_context_compaction.ipynb` — сжатие retrieved-контекста.

Качество:

| Режим | correctness | faithfulness | citations |
|---|---|---|---|
| full | 3.73 | 4.87 | 4.73 |
| compact | 3.50 | 4.73 | 4.63 |

Стоимость:

| Режим | context tokens | reduction |
|---|---|---|
| full | 1072.3 | — |
| compact | **497.7** | **52.1%** |

Сжатие уменьшает контекст на 52%, качество проседает умеренно. Формулоёмкие
разделы требуют более мягкого бюджета или formula-aware splitting.

`eval/notebooks/05_agent_vs_rag.ipynb` — сравнение RAG и LangGraph-агента
на тех же 30 вопросах.

Качество:

| Режим | correctness | faithfulness | citations | ответов со ссылками |
|---|---|---|---|---|
| RAG | 3.73 | **4.90** | **4.77** | 100% |
| агент | **4.63** | 4.53 | 4.60 | 100% |

Стоимость:

| Режим | latency mean | latency p50 | iterations | tool calls |
|---|---|---|---|---|
| RAG | 3.1 с | 3.1 с | 1 | 0 |
| агент | 7.9 с | 8.1 с | 2.40 | 2.03 |

Агент даёт +0.90 correctness за счёт нескольких поисков, но в 2.6 раза
медленнее и чаще добавляет детали, не подтверждённые пассажами. Tool-choice:
100% по набору и порядку инструментов. Подробности — в ноутбуке и `RESULTS.md`.

`eval/notebooks/06_ragas.ipynb` — RAGAS на 30 вопросах, RAG-режим.

| Метрика | Значение |
|---|---|
| faithfulness | **0.893** |
| answer relevancy | 0.782 |
| context precision | 0.578 |
| context recall | 0.675 |
| answer correctness | 0.520 |

RAGAS использует DeepSeek как судью и e5-small как эмбеддер. Ошибок — 0.
Главный вывод: ответы заземлены (faithfulness 0.893), но retrieval не всегда
достаёт нужные факты (context precision/recall 0.578/0.675). Пять вопросов
получили answer relevancy 0.0, потому что модель честно ответила «в контексте
этого нет» — это провал retrieval, а не генерации.

## Tool-choice метрики

`eval/golden/tool_qa.jsonl` — 6 вопросов, где важен выбор инструмента:

- `run_python` — вычисления;
- `search_wikipedia` — поиск по корпусу;
- комбинация `search_wikipedia` + `run_python` — multi-tool.

Метрики в `eval/tool_metrics.py`:

- `tool_choice_accuracy` — использован хотя бы один правильный инструмент;
- `exact_tool_set_accuracy` — набор инструментов совпал полностью;
- `tool_precision` / `tool_recall` / `tool_f1` — по множествам инструментов;
- `sequence_accuracy` — правильный порядок для multi-tool;
- `unnecessary_tool_rate` — доля лишних вызовов.

Фактические инструменты берутся из `AgentResponse.trace`.

Ответ:
- correctness / faithfulness / completeness по рубрике 1–5 (LLM-judge);
- citation precision / recall по `must_cite`;
- покрытие `key_facts`.

Агент:
- tool-choice accuracy;
- число итераций;
- latency;
- токены и стоимость.

Все замеры — в ноутбуках `eval/notebooks/`, результаты — в `eval/results/`.
