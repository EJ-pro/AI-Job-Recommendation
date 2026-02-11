export interface RoadmapStep {
    step: string;
    title: string;
    description: string;
    topics: string[];
    resources: { name: string; url: string }[];
    quiz?: {
        question: string;
        options: string[];
        correctAnswer: number;
    };
}

export interface TechStack {
    category: string;
    skills: string[];
}

export interface JobRole {
    id: string;
    title: string;
    description: string;
    long_description: string; // 상세 설명 (A-Z)
    salary_range: string; // 연봉 정보 (예: "초봉 4,000 ~ 5,000")
    difficulty: 'Easy' | 'Medium' | 'Hard' | 'Extreme';
    demand: 'Low' | 'Medium' | 'High' | 'Very High';
    responsibilities: string[]; // 주요 업무 리스트
    tech_stack: TechStack[]; // 기술 스택
    tags: string[];
    focus_areas: string[];
    roadmap: RoadmapStep[]; // 상세 로드맵
    faq: { question: string; answer: string }[]; // 자주 묻는 질문
}

export interface Question {
    id: number;
    question: string;
    options: {
        text: string;
        weights: Partial<Record<string, number>>;
    }[];
}

export const JOBS: JobRole[] = [
    {
        id: 'ai-app',
        title: 'AI Application Engineer',
        description: 'LLM API와 프레임워크를 활용해 실제 사용자가 쓰는 AI 서비스를 빠르게 개발합니다.',
        long_description: 'AI Application Engineer는 최신 AI 모델(LLM)을 활용하여 실제 사용자가 사용할 수 있는 웹/앱 서비스를 만드는 역할입니다. 모델 자체를 학습시키는 것보다, 잘 만들어진 모델(OpenAI, Claude 등)을 API로 호출하고, 이를 기존 시스템과 연결하여 가치를 창출하는 데 집중합니다. "AI를 활용한 풀스택 개발자"에 가깝습니다.',
        salary_range: '초봉 4,000 ~ 5,500만원',
        difficulty: 'Medium',
        demand: 'Very High',
        responsibilities: [
            'LangChain/LlamaIndex를 활용한 LLM 애플리케이션 개발',
            'RAG(검색 증강 생성) 파이프라인 구축 및 최적화',
            'FastAPI/Streamlit을 이용한 백엔드 및 데모 페이지 구현',
            'Prompt Engineering을 통한 답변 품질 개선'
        ],
        tech_stack: [
            { category: 'Language', skills: ['Python', 'TypeScript', 'JavaScript'] },
            { category: 'Framework', skills: ['FastAPI', 'Next.js', 'Streamlit', 'LangChain'] },
            { category: 'Database', skills: ['PostgreSQL', 'Pinecone (Vector DB)', 'Redis'] },
            { category: 'AI Tools', skills: ['OpenAI API', 'Hugging Face', 'Ollama'] }
        ],
        tags: ['Service', 'API', 'Fast-Paced'],
        focus_areas: [
            'LangChain/LlamaIndex 프레임워크 마스터',
            'Python 백엔드 (FastAPI/Streamlit)',
            'Prompt Engineering 기초'
        ],
        roadmap: [
            {
                step: 'Phase 1: AI 서비스 기초',
                title: 'LLM API 활용 및 챗봇 만들기',
                description: 'API를 통해 AI 모델과 대화하는 방법을 익히고, 간단한 챗봇 인터페이스를 구현합니다.',
                topics: ['REST API 호출', 'OpenAI Playground 사용법', 'Streamlit 기초', 'Python Asyncio'],
                resources: [
                    { name: 'OpenAI Quickstart', url: 'https://platform.openai.com/docs/quickstart' },
                    { name: 'Streamlit Documentation', url: 'https://docs.streamlit.io/' }
                ],
                quiz: {
                    question: 'OpenAI API를 사용할 때, 대화의 맥락(Context)을 유지하기 위해 보내야 하는 메시지 리스트의 역할(Role)이 아닌 것은?',
                    options: ['System', 'User', 'Assistant', 'Manager'],
                    correctAnswer: 3
                }
            },
            {
                step: 'Phase 2: 프레임워크 심화',
                title: 'LangChain & RAG 구현',
                description: '단순 대화를 넘어, 내 데이터를 참조하여 답변하는 RAG 시스템을 구축합니다.',
                topics: ['LangChain Components', 'Vector Database 원리', 'Embedding 개념', 'Chain & Agent'],
                resources: [
                    { name: 'LangChain Academy', url: 'https://python.langchain.com/docs/get_started/introduction' },
                    { name: 'Pinecone Learning Center', url: 'https://www.pinecone.io/learn/' }
                ]
            },
            {
                step: 'Phase 3: 실전 프로젝트',
                title: '나만의 AI 서비스 배포',
                description: '기획부터 배포까지 전체 과정을 경험하며 포트폴리오를 완성합니다.',
                topics: ['FastAPI 백엔드 구조화', 'Docker 배포', 'Vercel/Fly.io 호스팅', '서비스 모니터링'],
                resources: [
                    { name: 'Full Stack Deep Learning', url: 'https://fullstackdeeplearning.com/' }
                ]
            }
        ],
        faq: [
            { question: '수학을 잘해야 하나요?', answer: '아니요, 이 직무는 수학보다는 "구현력"과 "센스"가 중요합니다. API를 잘 다루고 사용자 경험(UX)을 고민하는 능력이 더 필요합니다.' },
            { question: '풀스택 개발자와 다른가요?', answer: '기본은 비슷하지만, LLM의 특성(Tokens, Context Window, Hallucination)을 이해하고 다루는 기술이 추가로 요구됩니다.' }
        ]
    },
    {
        id: 'prompt-eng',
        title: 'Prompt Engineer',
        description: '거대 언어 모델(LLM)의 잠재력을 극대화하기 위해 최적의 입력값(Prompt)을 설계하고 정제하는 전문가입니다.',
        long_description: 'Prompt Engineer는 "인공지능 소통 및 제어 전략가"입니다. 단순한 명령어 작성을 넘어, AI가 맥락에 맞는 답변을 생성하도록 논리적 추론 체계(Chain of Thought)를 구축합니다. RAG(검색 증강 생성) 기술을 활용해 외부 지식 데이터를 정확히 참조하도록 인터페이스를 구축하고, AI의 거짓 정보(Hallucination)를 제어하며 윤리적 가드레일을 설계합니다.',
        salary_range: '초봉 3,500 ~ 5,500만원',
        difficulty: 'Medium',
        demand: 'Very High',
        responsibilities: [
            'Prompt Strategy Design (CoT, ToT, ReAct)',
            'RAG (Retrieval-Augmented Generation) 파이프라인 최적화',
            'LLM 품질 검증 및 벤치마킹 (Evaluation)',
            'Hallucination 방지 및 윤리적 가드레일 설계',
            '프롬프트 자동화 및 시스템 통합 (LangChain)'
        ],
        tech_stack: [
            { category: 'LLM Models', skills: ['GPT-4', 'Claude 3', 'Gemini', 'Llama 3'] },
            { category: 'Frameworks', skills: ['LangChain', 'LlamaIndex', 'DSPy'] },
            { category: 'Data & DB', skills: ['Pinecone', 'ChromaDB', 'SQL', 'JSON'] },
            { category: 'DevOps', skills: ['Python', 'Docker', 'API (REST)', 'Git'] }
        ],
        tags: ['LLM', 'Creative', 'Logic', 'Engineering'],
        focus_areas: [
            'Advanced Prompting (Chain-of-Thought)',
            'RAG Architecture & Vector DB',
            'LLM Evaluation & Fine-tuning Basics'
        ],
        roadmap: [
            // 1. LLM Foundations
            {
                step: 'Phase 1: LLM 작동 원리 (Foundations)',
                title: 'How LLMs Work',
                description: '트랜스포머 아키텍처와 토큰(Token), 확률적 생성 원리를 이해합니다.',
                topics: ['Transformer Architecture', 'Tokenization & Embeddings', 'Temperature & Top-P', 'Context Window'],
                resources: [
                    { name: 'The Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/' }
                ],
                quiz: {
                    question: 'LLM이 텍스트를 생성할 때, 다음 단어를 선택하는 기준이 되는 확률 분포를 조절하여 창의성(무작위성)을 높이는 파라미터는?',
                    options: ['Temperature', 'Attention Mask', 'Batch Size', 'Learning Rate'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 프롬프트 기초 (Basic Prompting)',
                title: 'Prompting Basics',
                description: '명확하고 구체적인 지시를 통해 모델을 제어하는 기본 기법을 익힙니다.',
                topics: ['Zero-shot vs One-shot vs Few-shot', 'Role Prompting (Persona)', 'Instruction Formatting', 'Delimiters usage'],
                resources: [
                    { name: 'OpenAI Prompt Engineering Guide', url: 'https://platform.openai.com/docs/guides/prompt-engineering' }
                ],
                quiz: {
                    question: '모델에게 예시를 전혀 주지 않고 바로 작업을 수행하도록 요청하는 프롬프트 방식은?',
                    options: ['Zero-shot Prompting', 'Few-shot Prompting', 'Chain-of-Thought', 'Fine-tuning'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 3: 언어 모델 비교 분석',
                title: 'Model Comparison',
                description: 'GPT-4, Claude 3, Llama 3 등 주요 모델의 특성과 장단점을 파악합니다.',
                topics: ['OpenAI (GPT) vs Anthropic (Claude)', 'Open Source Models (Llama, Mistral)', 'Cost & Speed Latency', 'Reasoning Capabilities'],
                resources: [
                    { name: 'LLM Leaderboard (Hugging Face)', url: 'https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard' }
                ],
                quiz: {
                    question: '긴 문맥(Long Context) 처리에 강점이 있어, 책 한 권 분량의 텍스트 분석에 특히 유리한 모델 계열은?',
                    options: ['Claude (Anthropic)', 'BERT', 'T5', 'DALL-E'],
                    correctAnswer: 0
                }
            },
            // 2. Advanced Prompting Techniques
            {
                step: 'Phase 4: 생각의 사슬 (Chain of Thought)',
                title: 'Chain of Thought (CoT)',
                description: '복잡한 문제를 단계별로 추론하여 풀도록 유도하는 핵심 기법입니다.',
                topics: ['Zero-shot CoT ("Let\'s think step by step")', 'Manual CoT (Few-shot)', 'Math & Logic Reasoning', 'Hallucination Reduction'],
                resources: [
                    { name: 'CoT Paper', url: 'https://arxiv.org/abs/2201.11903' }
                ],
                quiz: {
                    question: '모델에게 "단계별로 생각해 보자(Let\'s think step by step)"라고 추가하여 추론 능력을 획기적으로 높이는 기법은?',
                    options: ['Zero-shot CoT', 'RAG', 'ReAct', 'Self-Consistency'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 5: 심화 추론 기법',
                title: 'Advanced Reasoning',
                description: '생각의 나무(ToT) 등 더 정교한 추론 프레임워크를 적용합니다.',
                topics: ['Tree of Thoughts (ToT)', 'Self-Consistency (Majority Voting)', 'Generated Knowledge Prompting', 'Least-to-Most Prompting'],
                resources: [
                    { name: 'Tree of Thoughts', url: 'https://arxiv.org/abs/2305.10601' }
                ],
                quiz: {
                    question: '여러 개의 추론 경로를 생성하고, 다수결(Majority Vote) 등을 통해 가장 신뢰할 수 있는 답을 선택하는 기법은?',
                    options: ['Self-Consistency', 'Zero-shot', 'Instruction Tuning', 'Embedding'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 6: 구조화된 출력 (Structured Output)',
                title: 'Structuring Outputs',
                description: 'AI의 답변을 JSON, XML 등 시스템이 처리 가능한 형식을 갖추도록 제어합니다.',
                topics: ['JSON Mode', 'Function Calling', 'Markdown Formatting', 'Output Parsers'],
                resources: [
                    { name: 'OpenAI Function Calling', url: 'https://platform.openai.com/docs/guides/function-calling' }
                ],
                quiz: {
                    question: 'LLM이 외부 API를 호출하거나 정형화된 데이터를 반환하도록, 사전에 정의된 함수 스키마를 JSON 형태로 출력하게 하는 기능은?',
                    options: ['Function Calling', 'Vector Database', 'RAG', 'Fine-tuning'],
                    correctAnswer: 0
                }
            },
            // 3. Technical Implementation & Data
            {
                step: 'Phase 7: 임베딩과 벡터 DB',
                title: 'Embeddings & Vector Databases',
                description: '텍스트를 숫자로 변환(Vector)하여 의미론적 검색을 수행하는 원리를 배웁니다.',
                topics: ['Vector Embeddings', 'Cosine Similarity', 'Pinecone / ChromaDB / Weaviate', 'Semantic Search'],
                resources: [
                    { name: 'Pinecone Learning Center', url: 'https://www.pinecone.io/learn/' }
                ],
                quiz: {
                    question: '두 벡터(문장) 사이의 유사도를 측정할 때 가장 널리 사용되는 지표로, 두 벡터 사이의 각도를 기반으로 하는 것은?',
                    options: ['Cosine Similarity (코사인 유사도)', 'Euclidean Distance', 'Manhattan Distance', 'Jaccard Similarity'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 8: RAG 기초 (Retrieval-Augmented Generation)',
                title: 'RAG Fundamentals',
                description: 'LLM이 학습하지 않은 최신 정보나 사내 데이터를 참조하여 답변하도록 만듭니다.',
                topics: ['Document Loading & Splitting', 'Retrieval Strategies', 'Context Injection', 'Handling Context Limits'],
                resources: [
                    { name: 'LangChain RAG Tutorial', url: 'https://python.langchain.com/docs/use_cases/question_answering/' }
                ],
                quiz: {
                    question: 'LLM의 환각(Hallucination)을 줄이고 최신 정보를 반영하기 위해, 외부 지식 베이스에서 관련 문서를 검색하여 프롬프트에 포함시키는 기술은?',
                    options: ['RAG (검색 증강 생성)', 'Fine-tuning', 'Pre-training', 'RLHF'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 9: 고급 RAG 기법',
                title: 'Advanced RAG',
                description: '검색 정확도를 높이고 답변 품질을 최적화하는 고급 RAG 시스템을 구축합니다.',
                topics: ['Hybrid Search (Keyword + Semantic)', 'Re-ranking', 'Parent Document Retriever', 'HyDE (Hypothetical Document Embeddings)'],
                resources: [
                    { name: 'Advanced RAG Techniques', url: 'https://www.deeplearning.ai/' }
                ],
                quiz: {
                    question: '1차 검색된 문서들의 순위를 다시 매겨(Re-ranking), 가장 관련성 높은 문서를 상위에 배치하여 LLM에게 전달하는 기술은?',
                    options: ['Re-ranking', 'Embedding', 'Clustering', 'Tokenization'],
                    correctAnswer: 0
                }
            },
            // 4. Frameworks & Tools
            {
                step: 'Phase 10: 랭체인 (LangChain)',
                title: 'LangChain Mastery',
                description: 'LLM 애플리케이션 개발의 표준 프레임워크인 LangChain을 깊이 있게 다룹니다.',
                topics: ['Chains & LCEL', 'Memory Management', 'Document Loaders', 'Custom Tools'],
                resources: [
                    { name: 'LangChain Documentation', url: 'https://python.langchain.com' }
                ],
                quiz: {
                    question: 'LangChain에서 여러 컴포넌트(프롬프트, 모델, 출력 파서 등)를 선언적으로 연결하여 파이프라인을 구성하는 문법은?',
                    options: ['LCEL (LangChain Expression Language)', 'SQL', 'RegEx', 'Gremlin'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 라마인덱스 (LlamaIndex)',
                title: 'Data Framework (LlamaIndex)',
                description: '데이터 중심의 LLM 앱 개발을 위한 LlamaIndex를 익힙니다.',
                topics: ['Data Connectors', 'Index Structures', 'Query Engines', 'Router'],
                resources: [
                    { name: 'LlamaIndex Docs', url: 'https://docs.llamaindex.ai/' }
                ],
                quiz: {
                    question: 'LlamaIndex에서 비정형 데이터를 LLM이 이해하기 쉬운 구조(Index)로 변환하고, 이를 쿼리할 수 있게 돕는 핵심 역할은?',
                    options: ['Indexing & Querying', 'Training', 'Fine-tuning', 'Reinforcement Learning'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 12: 프롬프트 최적화 도구 (DSPy)',
                title: 'DSPy & Prompt Optimization',
                description: '수동 프롬프팅을 넘어, 프로그래밍 방식으로 프롬프트를 최적화하는 DSPy를 배웁니다.',
                topics: ['Declarative Self-improving Language Models', 'Signatures & Modules', 'Teleprompters (Optimizers)', 'Automated Prompt Tuning'],
                resources: [
                    { name: 'DSPy GitHub', url: 'https://github.com/stanfordnlp/dspy' }
                ],
                quiz: {
                    question: '스탠포드에서 개발한 프레임워크로, 프롬프트를 하드코딩하는 대신 "Signature"로 입출력을 정의하면 최적의 프롬프트를 자동으로 찾아내는 도구는?',
                    options: ['DSPy', 'PyTorch', 'TensorFlow', 'React'],
                    correctAnswer: 0
                }
            },
            // 5. Build Agents
            {
                step: 'Phase 13: 에이전트 기초 (Agents)',
                title: 'Building AI Agents',
                description: '스스로 도구를 선택하고 행동을 결정하는 AI 에이전트를 만듭니다.',
                topics: ['ReAct Pattern', 'Tool Use (Function Calling)', 'Agent Executors', 'Planning'],
                resources: [
                    { name: 'ReAct Paper', url: 'https://arxiv.org/abs/2210.03629' }
                ],
                quiz: {
                    question: 'LLM이 "생각(Reasoning)"과 "행동(Acting)"을 번갈아 수행하며 도구를 사용하여 문제를 해결하는 프롬프트 패턴은?',
                    options: ['ReAct', 'CoT', 'Few-shot', 'RAG'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 14: 멀티 에이전트 시스템',
                title: 'Multi-Agent Systems',
                description: '여러 AI 에이전트가 협력하여 복잡한 과업을 수행하는 시스템을 구축합니다.',
                topics: ['LangGraph', 'AutoGPT / BabyAGI', 'CrewAI', 'Role-based Collaboration'],
                resources: [
                    { name: 'LangGraph Tutorials', url: 'https://github.com/langchain-ai/langgraph' }
                ],
                quiz: {
                    question: '여러 개의 특화된 에이전트(예: 연구원, 작가, 검수자)가 서로 대화하며 작업을 수행하는 프레임워크는?',
                    options: ['Multi-Agent System (e.g., CrewAI)', 'Single Agent', 'RAG', 'Vector DB'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 15: 자율 운영 (Autonomous AI)',
                title: 'Autonomous Operations',
                description: 'BabyAGI, AutoGPT와 같이 목표만 주어지면 스스로 하위 태스크를 생성하고 실행하는 원리를 파악합니다.',
                topics: ['Task Queues & Prioritization', 'Memory Management (Short/Long-term)', 'Self-Correction', 'Human-in-the-loop'],
                resources: [
                    { name: 'AutoGPT GitHub', url: 'https://github.com/Significant-Gravitas/Auto-GPT' }
                ],
                quiz: {
                    question: '자율 에이전트가 무한 루프에 빠지거나 엉뚱한 행동을 하는 것을 방지하기 위해, 중간에 사람이 개입하여 피드백을 주는 방식은?',
                    options: ['Human-in-the-loop', 'Reinforcement Learning', 'Unsupervised Learning', 'Zero-shot'],
                    correctAnswer: 0
                }
            },
            // 6. Evaluation & OPS
            {
                step: 'Phase 16: 프롬프트 평가 (Evaluation)',
                title: 'Prompt Evaluation (Evals)',
                description: '주관적인 "좋아 보인다"가 아닌, 정량적인 지표로 프롬프트 성능을 측정합니다.',
                topics: ['LLM-as-a-Judge', 'Ragas (RAG Evaluation)', 'LangSmith Tracing', 'G-Eval'],
                resources: [
                    { name: 'Ragas Documentation', url: 'https://docs.ragas.io/' }
                ],
                quiz: {
                    question: 'RAG 파이프라인의 성능을 평가할 때, "검색된 문서가 질문과 얼마나 관련이 있는가"를 측정하는 지표는?',
                    options: ['Context Relevance', 'Faithfulness', 'Answer Correctness', 'Latency'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 17: 프롬프트 보안 (Security)',
                title: 'Prompt Injection & Security',
                description: 'AI 모델을 속이려는 공격을 이해하고 방어하는 기법을 배웁니다.',
                topics: ['Prompt Injection', 'Jailbreaking (DAN mode)', 'Input Validation', 'Plaque / Guardrails'],
                resources: [
                    { name: 'OWASP Top 10 for LLM', url: 'https://owasp.org/www-project-top-10-for-large-language-model-applications/' }
                ],
                quiz: {
                    question: '악의적인 사용자가 프롬프트에 숨겨진 명령어를 주입하여, 모델이 원래 지침을 무시하고 해로운 동작을 하게 만드는 공격은?',
                    options: ['Prompt Injection', 'SQL Injection', 'DDOS', 'Phishing'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 18: 비용 및 성능 최적화',
                title: 'Cost & Latency Optimization',
                description: '토큰 사용량을 줄이고 응답 속도를 높이는 실무적인 최적화 기술입니다.',
                topics: ['Prompt Compression', 'Caching (Semantic Cache)', 'Model Selection Strategy', 'Streaming Responses'],
                resources: [
                    { name: 'LiteLLM Proxy', url: 'https://docs.litellm.ai/' }
                ],
                quiz: {
                    question: '이전에 동일하거나 유사한 질문이 들어왔을 때, LLM을 호출하지 않고 저장된 답변을 바로 반환하여 비용과 시간을 아끼는 기술은?',
                    options: ['Semantic Caching', 'Compression', 'Shard', 'Batching'],
                    correctAnswer: 0
                }
            },
            // 7. Domain & Career
            {
                step: 'Phase 19: 파인튜닝 기초 (Fine-tuning)',
                title: 'Prompt Engineering vs Fine-tuning',
                description: '프롬프트만으로 부족할 때, 모델 자체를 미세조정하는 시점과 방법을 이해합니다.',
                topics: ['Instruction Tuning', 'PEFT (LoRA)', 'Data Formatting (JSONL)', 'When to Fine-tune'],
                resources: [
                    { name: 'Brex\'s Prompt Engineering vs Fine-tuning', url: 'https://github.com/brexhq/prompt-engineering' }
                ],
                quiz: {
                    question: '프롬프트 엔지니어링의 한계(토큰 제한, 복잡한 스타일 모방 불가 등)를 극복하기 위해, 데이터셋으로 모델 가중치를 업데이트하는 방법은?',
                    options: ['Fine-tuning', 'RAG', 'Embedding', 'Inferencing'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 20: 윤리와 규제 (Ethics)',
                title: 'AI Ethics & Responsibility',
                description: '편향성(Bias), 저작권, 개인정보 보호 등 AI 윤리 문제를 다룹니다.',
                topics: ['Bias Mitigation', 'Copyright Issues', 'PII (개인정보) Masking', 'Responsible AI'],
                resources: [
                    { name: 'Google Responsible AI Practices', url: 'https://ai.google/responsibility/principles/' }
                ],
                quiz: {
                    question: 'AI 모델이 학습 데이터의 편향을 그대로 반영하여 특정 인종이나 성별에 대해 차별적인 발언을 하는 것을 방지하는 작업은?',
                    options: ['Bias Mitigation (편향 완화)', 'Data Augmentation', 'Overfitting', 'Pruning'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 21: 포트폴리오 (Portfolio)',
                title: 'Building a Portfolio',
                description: '실제 문제를 해결한 프롬프트 사례집과 AI 앱 데모를 준비합니다.',
                topics: ['Prompt Library 구축', 'LangChain 데모 앱', 'Technical Blog', 'Hackathons'],
                resources: [
                    { name: 'Vercel AI SDK', url: 'https://sdk.vercel.ai/docs' }
                ],
                quiz: {
                    question: '프롬프트 엔지니어의 포트폴리오로 가장 적절하지 않은 것은?',
                    options: ['단순히 "ChatGPT 써봤음"이라고 적는 것', '해결한 문제와 개선된 성능 지표(Before/After) 제시', '직접 개발한 챗봇 데모 링크', '작성한 프롬프트 템플릿 공유'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '개발 지식이 없어도 되나요?', answer: '초기에는 괜찮지만, RAG나 에이전트 구축 등 고급 업무를 위해서는 Python과 API 활용 능력이 필수적입니다.' },
            { question: '미래에도 유망한가요?', answer: '단순한 "질문 작성"은 사라지겠지만, 복잡한 시스템을 지휘하고 AI를 평가/검증하는 "AI 오케스트레이터"로서의 역할은 더욱 커질 것입니다.' }
        ]
    },
    {
        id: 'mlops',
        title: 'MLOps Engineer',
        description: '머신러닝 모델의 개발부터 배포, 운영, 모니터링까지 전 과정을 자동화하고 최적화하는 파이프라인 전문가입니다.',
        long_description: 'MLOps Engineer는 "Machine Learning"과 "Operations"의 합성어로, 안정적이고 효율적인 머신러닝 시스템을 구축하는 역할을 합니다. 데이터 사이언티스트가 만든 모델을 실제 서비스에 적용하기 위해 필요한 인프라(AWS/GCP), CI/CD 파이프라인, 모델 서빙, 모니터링 시스템을 설계하고 운영합니다. 코드 품질 관리부터 모델 성능 모니터링까지 AI 서비스의 라이프사이클 전체를 책임집니다.',
        salary_range: '초봉 4,500 ~ 6,500만원',
        difficulty: 'Hard',
        demand: 'High',
        responsibilities: [
            'End-to-End ML Pipeline 구축 및 자동화 (Airflow, Kubeflow)',
            '모델 서빙 인프라 설계 및 운영 (Kubernetes, Docker)',
            '실험 추적 및 모델 레지스트리 관리 (MLflow, Weights & Biases)',
            '데이터 및 모델 품질 모니터링 (Drift Detection)'
        ],
        tech_stack: [
            { category: 'Cloud & Infrastructure', skills: ['AWS/GCP/Azure', 'Docker', 'Kubernetes', 'Terraform'] },
            { category: 'MLOps Tools', skills: ['MLflow', 'Kubeflow', 'Airflow', 'Jenkins/GitHub Actions'] },
            { category: 'Serving', skills: ['FastAPI', 'TorchServe', 'Triton Inference Server'] },
            { category: 'Monitoring', skills: ['Prometheus', 'Grafana', 'Evidently AI'] }
        ],
        tags: ['Infrastructure', 'Cloud', 'Pipeline', 'Automation'],
        focus_areas: [
            'Container Orchestration (K8s)',
            'Continuous Integration/Deployment (CI/CD)',
            'Model Serving Optimization',
            'Infrastructure as Code (IaC)'
        ],
        roadmap: [
            // 1. Foundations
            {
                step: 'Phase 1: 개발 환경 & 버전 관리',
                title: 'Development Environment',
                description: '효율적인 협업과 재현성을 위해 리눅스, Git, Python 환경을 완벽하게 세팅합니다.',
                topics: ['Linux/Bash Scripting', 'Git Flow & GitHub', 'Python Virtual Environments (Poetry/Conda)', 'IDE Setup (VS Code)'],
                resources: [
                    { name: 'Missing Semester (MIT)', url: 'https://missing.csail.mit.edu/' }
                ],
                quiz: {
                    question: 'Git에서 여러 브랜치의 변경 사항을 병합할 때, 커밋 히스토리를 깔끔하게 유지하기 위해 사용하는 명령어옵션은?',
                    options: ['git rebase', 'git merge --no-ff', 'git checkout', 'git stash'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 컨테이너 가상화 (Docker)',
                title: 'Containerization',
                description: '어떤 환경에서도 동일하게 실행되는 "컨테이너" 기술을 마스터합니다.',
                topics: ['Docker Architecture', 'Dockerfile Optimization', 'Multi-stage Builds', 'Docker Compose'],
                resources: [
                    { name: 'Docker for Beginners', url: 'https://docker-curriculum.com/' }
                ],
                quiz: {
                    question: 'Docker 컨테이너가 종료되어도 데이터를 영구적으로 저장하기 위해 사용하는 기능은?',
                    options: ['Docker Volume', 'Docker Network', 'Docker Image', 'Docker Socket'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 3: 컨테이너 오케스트레이션 (Kubernetes)',
                title: 'Kubernetes Fundamentals',
                description: '수많은 컨테이너를 관리하고 스케일링하기 위한 사실상의 표준, K8s를 배웁니다.',
                topics: ['Pods, Services, Deployments', 'ConfigMaps & Secrets', 'Helm Charts', 'Ingress Controller'],
                resources: [
                    { name: 'Kubernetes Docs', url: 'https://kubernetes.io/docs/home/' }
                ],
                quiz: {
                    question: 'Kubernetes에서 애플리케이션의 설정 정보(DB 주소 등)를 코드와 분리하여 저장하는 리소스 객체는?',
                    options: ['ConfigMap', 'Pod', 'Service', 'PersistentVolume'],
                    correctAnswer: 0
                }
            },
            // 2. Continuous X (CI/CD/CT)
            {
                step: 'Phase 4: CI/CD 파이프라인',
                title: 'Continuous Integration/Deployment',
                description: '코드 변경 사항을 자동으로 테스트하고 배포하는 자동화 파이프라인을 구축합니다.',
                topics: ['GitHub Actions', 'Jenkins/GitLab CI', 'Automated Testing (PyTest)', 'Linting & Formatting'],
                resources: [
                    { name: 'GitHub Actions Documentation', url: 'https://docs.github.com/en/actions' }
                ],
                quiz: {
                    question: 'CI/CD 파이프라인에서 코드가 푸시될 때마다 자동으로 실행되어야 하는 가장 기본적인 단계는?',
                    options: ['Unit Test & Build', 'Deploy to Production', 'Model Training', 'Data Ingestion'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 5: 머신러닝 파이프라인 (Orchestration)',
                title: 'Workflow Orchestration',
                description: '데이터 전처리부터 학습까지 복잡한 워크플로우를 스케줄링하고 관리합니다.',
                topics: ['Apache Airflow', 'Kubeflow Pipelines', 'Task Dependencies (DAGs)', 'Workflow Monitoring'],
                resources: [
                    { name: 'Apache Airflow Tutorial', url: 'https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html' }
                ],
                quiz: {
                    question: 'Airflow에서 작업(Task)들의 실행 순서와 의존성을 정의하는 그래프 구조를 무엇이라 하는가?',
                    options: ['DAG (Directed Acyclic Graph)', 'Tree', 'Queue', 'Stack'],
                    correctAnswer: 0
                }
            },
            // 3. Model Tracking & Management
            {
                step: 'Phase 6: 실험 추적 (Experiment Tracking)',
                title: 'Tracking Experiments',
                description: '모델의 하이퍼파라미터, 성능 지표, 아티팩트를 체계적으로 기록하고 비교합니다.',
                topics: ['MLflow Tracking', 'Weights & Biases (W&B)', 'Metrics Logging', 'Artifact Storage'],
                resources: [
                    { name: 'MLflow Documentation', url: 'https://mlflow.org/docs/latest/index.html' }
                ],
                quiz: {
                    question: 'MLflow Tracking 서버에 기록되지 않는 정보는?',
                    options: ['Source Code Line-by-Line Execution', 'Parameters', 'Metrics', 'Artifacts (Model files)'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 7: 모델 레지스트리 (Model Registry)',
                title: 'Model Versioning',
                description: '학습된 모델의 버전을 관리하고, Staging/Production 단계로 승격시키는 프로세스를 익힙니다.',
                topics: ['Model Versioning', 'Model Staging', 'Model Lineage', 'Release Management'],
                resources: [
                    { name: 'MLflow Model Registry', url: 'https://mlflow.org/docs/latest/model-registry.html' }
                ],
                quiz: {
                    question: '모델 레지스트리에서 검증이 완료된 모델을 서비스 가능한 상태로 표시하는 태그나 단계(Stage)는?',
                    options: ['Production', 'Archived', 'Development', 'N/A'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 8: 피처 스토어 (Feature Store)',
                title: 'Feature Engineering Platform',
                description: '학습과 추론 시점에 동일한 데이터를 제공하여 Training-Serving Skew를 방지합니다.',
                topics: ['Online vs Offline Stores', 'Feast (Feature Store)', 'Feature Sharing', 'Point-in-Time Correctness'],
                resources: [
                    { name: 'Feast Documentation', url: 'https://docs.feast.dev/' }
                ],
                quiz: {
                    question: '피처 스토어(Feature Store)의 핵심 기능 중 하나로, 과거 특정 시점의 데이터 상태를 정확히 조회하는 기능은?',
                    options: ['Point-in-Time (Time Travel) Query', 'Data Augmentation', 'Real-time Streaming', 'Batch Processing'],
                    correctAnswer: 0
                }
            },
            // 4. Model Serving & Deployment
            {
                step: 'Phase 9: 모델 서빙 API',
                title: 'Model Serving Basics',
                description: '학습된 모델을 REST API나 gRPC 형태로 패키징하여 외부에서 호출할 수 있게 합니다.',
                topics: ['FastAPI', 'Flask', 'Serialization (Pickle/ONNX)', 'API Documentation (Swagger)'],
                resources: [
                    { name: 'FastAPI Tutorial', url: 'https://fastapi.tiangolo.com/tutorial/' }
                ],
                quiz: {
                    question: 'Python 비동기(ASGI) 프레임워크로, 높은 성능과 자동 문서화를 제공하여 모델 서빙에 자주 쓰이는 것은?',
                    options: ['FastAPI', 'Flask', 'Django', 'Bottle'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 10: 고성능 서빙 엔진',
                title: 'Advanced Model Serving',
                description: '대량의 트래픽 처리를 위해 최적화된 전문 서빙 프레임워크를 사용합니다.',
                topics: ['TorchServe / TensorFlow Serving', 'NVIDIA Triton Inference Server', 'Dynamic Batching', 'Model Ensemble'],
                resources: [
                    { name: 'Triton Inference Server', url: 'https://developer.nvidia.com/nvidia-triton-inference-server' }
                ],
                quiz: {
                    question: '여러 개의 추론 요청을 모아서 한 번에 처리하여 GPU 활용률과 처리량(Throughput)을 높이는 기법은?',
                    options: ['Dynamic Batching', 'Model Pruning', 'Quantization', 'Caching'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 배포 전략 (Deployment Strategies)',
                title: 'Safe Deployment',
                description: '서비스 중단 없이 안전하게 새로운 모델을 배포하는 전략을 배웁니다.',
                topics: ['Blue-Green Deployment', 'Canary Release', 'A/B Testing', 'Shadow Deployment'],
                resources: [
                    { name: 'Deployment Strategies on K8s', url: 'https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#strategy' }
                ],
                quiz: {
                    question: '새 버전의 모델을 일부 사용자에게만 노출시켜 안정성을 검증한 후 점진적으로 트래픽을 늘리는 배포 방식은?',
                    options: ['Canary Release', 'Blue-Green Deployment', 'Recreate', 'Big Bang'],
                    correctAnswer: 0
                }
            },
            // 5. Monitoring & Observability
            {
                step: 'Phase 12: 시스템 모니터링',
                title: 'Infrastructure Monitoring',
                description: '서버의 CPU, Memory, GPU 사용량과 응답 속도 등을 실시간으로 감시합니다.',
                topics: ['Prometheus (Metrics Collection)', 'Grafana (Visualization)', 'Alert Manager', 'Log Aggregation (ELK/Loki)'],
                resources: [
                    { name: 'Prometheus Basics', url: 'https://prometheus.io/docs/introduction/overview/' }
                ],
                quiz: {
                    question: '시계열 데이터(Time Series Data)를 수집하고 쿼리하는 데 특화된 오픈소스 모니터링 시스템은?',
                    options: ['Prometheus', 'MySQL', 'Redis', 'MongoDB'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 13: 모델 성능 모니터링',
                title: 'Model Observability',
                description: '모델의 예측 성능이 시간이 지남에 따라 저하되는지(Drift)를 감지합니다.',
                topics: ['Data Drift vs Concept Drift', 'Evidently AI / Arize', 'Outlier Detection', 'Model Fairness'],
                resources: [
                    { name: 'Evidently AI', url: 'https://www.evidentlyai.com/' }
                ],
                quiz: {
                    question: '입력 데이터의 분포(P(X))가 학습 시점과 달라져서 모델 성능이 떨어지는 현상은?',
                    options: ['Covariate Shift (Data Drift)', 'Concept Drift', 'Label Shift', 'Prior Probability Shift'],
                    correctAnswer: 0
                }
            },
            // 6. Infrastructure & Cloud
            {
                step: 'Phase 14: 클라우드 (Cloud Providers)',
                title: 'Cloud ML Platforms',
                description: 'AWS, GCP 등 클라우드 벤더가 제공하는 매니지드 ML 서비스를 활용합니다.',
                topics: ['AWS SageMaker', 'GCP Vertex AI', 'Azure ML', 'Serverless Inference (Lambda)'],
                resources: [
                    { name: 'AWS SageMaker', url: 'https://aws.amazon.com/sagemaker/' }
                ],
                quiz: {
                    question: 'AWS에서 제공하는 완전 관리형 머신러닝 서비스로, 빌드/학습/배포를 통합 제공하는 플랫폼은?',
                    options: ['SageMaker', 'EC2', 'Lambda', 'Fargate'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 15: IaC (Infrastructure as Code)',
                title: 'Infrastructure Automation',
                description: '클라우드 인프라를 코드로 정의하여 생성, 변경, 삭제를 자동화합니다.',
                topics: ['Terraform', 'Ansible', 'Pulumi', 'State Management'],
                resources: [
                    { name: 'Terraform Introduction', url: 'https://developer.hashicorp.com/terraform/intro' }
                ],
                quiz: {
                    question: 'Terraform에서 인프라의 현재 상태를 저장하고 실제 리소스와 매핑하는 파일은?',
                    options: ['State File (.tfstate)', 'Config File', 'Log File', 'Manifest'],
                    correctAnswer: 0
                }
            },
            // 7. Data Engineering for MLOps
            {
                step: 'Phase 16: 데이터 처리 (Data Processing)',
                title: 'Big Data Processing',
                description: '대용량 데이터를 효율적으로 처리하기 위한 분산 처리 프레임워크를 익힙니다.',
                topics: ['Apache Spark', 'Dask', 'Data Lakes (S3/GCS)', 'Extract-Transform-Load (ETL)'],
                resources: [
                    { name: 'PySpark Documentation', url: 'https://spark.apache.org/docs/latest/api/python/' }
                ],
                quiz: {
                    question: '메모리 기반의 분산 데이터 처리 프레임워크로, 하둡(Hadoop)보다 훨씬 빠른 속도를 제공하는 것은?',
                    options: ['Apache Spark', 'Hive', 'Pig', 'MapReduce'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 17: 데이터 버전 관리 (DVC)',
                title: 'Data Version Control',
                description: '코드뿐만 아니라 대용량 데이터셋의 변경 사항도 버전 관리합니다.',
                topics: ['DVC (Data Version Control)', 'Pachyderm', 'Data Lineage', 'Reproducibility'],
                resources: [
                    { name: 'DVC.org', url: 'https://dvc.org/' }
                ],
                quiz: {
                    question: 'Git과 유사한 명령어를 사용하며, 실제 대용량 데이터는 S3 등에 저장하고 메타데이터만 Git으로 관리하는 도구는?',
                    options: ['DVC', 'Git LFS', 'SVN', 'Mercurial'],
                    correctAnswer: 0
                }
            },
            // 8. Security & Governance
            {
                step: 'Phase 18: 보안과 거버넌스',
                title: 'ML Security & Governance',
                description: '모델의 보안 취약점을 방어하고, 윤리적/법적 규제를 준수합니다.',
                topics: ['Adversarial Attacks', 'Model Explainability (XAI)', 'RBAC (Role-Based Access Control)', 'GDPR & Compliance'],
                resources: [
                    { name: 'Adversarial Machine Learning', url: 'https://github.com/dberkholz/adversarial-machine-learning' }
                ],
                quiz: {
                    question: '입력 데이터에 미세한 노이즈를 섞어 AI 모델을 오작동하게 만드는 공격 기법은?',
                    options: ['Adversarial Attack (적대적 공격)', 'DDoS', 'SQL Injection', 'Man-in-the-Middle'],
                    correctAnswer: 0
                }
            },
            // 9. Advanced Trends
            {
                step: 'Phase 19: LLM Ops',
                title: 'Large Language Model Ops',
                description: '거대 언어 모델(LLM)을 효율적으로 튜닝하고 서빙하기 위한 특화된 운영 기술입니다.',
                topics: ['Fine-tuning (PEFT)', 'Vector Databases', 'Prompt Management', 'LLM Evaluation'],
                resources: [
                    { name: 'Full Stack LLM Bootcamp', url: 'https://fullstackdeeplearning.com/llm-bootcamp/' }
                ],
                quiz: {
                    question: 'LLM의 출력을 제어하고 최적화하기 위해 입력 텍스트(Prompt)를 체계적으로 관리하고 버전 관리하는 기술은?',
                    options: ['Prompt Engineering & Management', 'Feature Engineering', 'Hyperparameter Tuning', 'Data Cleaning'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 20: 엣지 MLOps (Edge AI)',
                title: 'MLOps on Edge',
                description: '리소스가 제한된 엣지 디바이스에 모델을 배포하고 업데이트하는기술입니다.',
                topics: ['OTA (Over-The-Air) Updates', 'Model Compression', 'TensorLite / Edge TPU', 'Fleet Management'],
                resources: [
                    { name: 'Edge Impulse', url: 'https://www.edgeimpulse.com/' }
                ],
                quiz: {
                    question: '원격지에 있는 수많은 엣지 디바이스의 펌웨어나 모델을 무선으로 업데이트하는 기술은?',
                    options: ['OTA (Over-The-Air)', 'SSH', 'FTP', 'USB'],
                    correctAnswer: 0
                }
            },
            // 10. Career
            {
                step: 'Phase 21: 커리어 & 프로젝트',
                title: 'End-to-End Project',
                description: '문제 정의부터 배포, 모니터링까지 전 과정을 아우르는 MLOps 프로젝트를 완성합니다.',
                topics: ['System Design Interview', 'Open Source Contribution', 'Tech Blog', 'Portfolio Building'],
                resources: [
                    { name: 'Made With ML', url: 'https://madewithml.com/' }
                ],
                quiz: {
                    question: 'MLOps 프로젝트 포트폴리오에서 가장 강조해야 할 역량은?',
                    options: ['전체 파이프라인의 자동화 및 문제 해결 과정', '가장 복잡한 모델 사용', 'UI 디자인', '데이터의 양'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '신입도 가능한가요?', answer: '진입 장벽이 높은 편입니다. 백엔드나 시스템 엔지니어로 시작해서 ML 관련 경험을 쌓고 넘어오는 경우가 많습니다.' },
            { question: '수학적 지식이 필요한가요?', answer: '모델의 내부 원리보다는, 모델이 실행되는 "환경"과 "시스템"에 대한 이해가 훨씬 중요합니다.' }
        ]
    },
    {
        id: 'data-eng',
        title: 'Data Engineer',
        description: '대규모 데이터를 수집, 저장, 처리 및 관리하는 파이프라인과 아키텍처를 설계하는 데이터 인프라 전문가입니다.',
        long_description: 'Data Engineer는 데이터 사이언티스트와 ML 엔지니어가 작업할 수 있는 "데이터 고속도로"를 건설합니다. 원천 데이터(Raw Data)를 다양한 소스에서 추출(Extract)하고, 비즈니스 로직에 맞게 변환(Transform)하여, 데이터 웨어하우스나 호수(Lake)에 적재(Load)하는 ETL/ELT 파이프라인을 구축합니다. 또한 데이터의 품질(Validation), 보안(Governance), 흐름(Lineage)을 관리하여 "믿을 수 있는 데이터"를 제공하는 것이 핵심 미션입니다.',
        salary_range: '초봉 4,000 ~ 6,000만원',
        difficulty: 'Medium',
        demand: 'Very High',
        responsibilities: [
            'ETL/ELT 파이프라인 설계 및 운영 (Airflow, dbt)',
            '대용량 데이터 분산 처리 시스템 구축 (Spark, Hadoop)',
            '데이터 웨어하우스/레이크 모델링 (Snowflake, BigQuery)',
            '실시간 데이터 스트리밍 처리 (Kafka, Flink)'
        ],
        tech_stack: [
            { category: 'Language', skills: ['Python', 'SQL (Advanced)', 'Scala', 'Java'] },
            { category: 'Compute', skills: ['Apache Spark', 'Databricks', 'Hadoop MapReduce'] },
            { category: 'Orchestration', skills: ['Apache Airflow', 'Prefect', 'Dagster', 'dbt'] },
            { category: 'Storage & Warehouse', skills: ['Snowflake', 'BigQuery', 'Redshift', 'S3', 'HDFS'] }
        ],
        tags: ['BigData', 'Infrastructure', 'ETL', 'Architecture'],
        focus_areas: [
            'Advanced SQL & Data Modeling',
            'Distributed Computing (Spark)',
            'Workflow Orchestration (Airflow)',
            'Cloud Data Warehousing'
        ],
        roadmap: [
            // 1. Fundamentals
            {
                step: 'Phase 1: CS 기초 & 리눅스',
                title: 'CS Basics & Linux',
                description: '터미널 환경에서 대용량 파일을 다루고 자동화 스크립트를 작성하는 능력을 기릅니다.',
                topics: ['Linux/Bash Commands', 'Shell Scripting', 'SSH & Network Basics', 'File Systems (HDFS 기초)'],
                resources: [
                    { name: 'Linux Command Line Basics', url: 'https://ubuntu.com/tutorials/command-line-for-beginners' }
                ],
                quiz: {
                    question: '리눅스에서 대용량 로그 파일의 마지막 100줄을 실시간으로 계속 확인하고 싶을 때 사용하는 명령어는?',
                    options: ['tail -f filename.log', 'cat filename.log', 'head -n 100 filename.log', 'grep filename.log'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 프로그래밍 언어 (Python/SQL)',
                title: 'Programming skills',
                description: '데이터 조작을 위한 Python과 복잡한 쿼리 작성을 위한 SQL을 마스터합니다.',
                topics: ['Python Data Structures', 'Pandas/Polars', 'Functional Programming', 'Advanced SQL (Optimization)'],
                resources: [
                    { name: 'LeetCode Database Problems', url: 'https://leetcode.com/problemset/database/' }
                ],
                quiz: {
                    question: 'SQL에서 윈도우 함수(Window Function)를 사용하여, 부서별로 급여 순위를 매길 때 사용하는 구문은?',
                    options: ['RANK() OVER (PARTITION BY dept ORDER BY salary)', 'GROUP BY dept ORDER BY salary', 'SELECT * FROM salary WHERE dept=...', 'JOIN dept ON salary'],
                    correctAnswer: 0
                }
            },
            // 2. Database & Modeling
            {
                step: 'Phase 3: 관계형 데이터베이스 (RDBMS)',
                title: 'Relational Databases',
                description: '데이터의 정합성을 보장하는 RDBMS의 내부 동작 원리와 설계를 배웁니다.',
                topics: ['PostgreSQL/MySQL', 'ACID Transaction', 'Normalization Forms', 'Indexing Strategy'],
                resources: [
                    { name: 'CMU Database Systems Course', url: 'https://15445.courses.cs.cmu.edu/' }
                ],
                quiz: {
                    question: '데이터베이스에서 트랜잭션이 안전하게 수행된다는 것을 보장하는 성질인 ACID에 포함되지 않는 것은?',
                    options: ['Atomicity (원자성)', 'Consistency (일관성)', 'Isolation (고립성)', 'Dependency (의존성)'],
                    correctAnswer: 3
                }
            },
            {
                step: 'Phase 4: 데이터 모델링 & NoSQL',
                title: 'Data Modeling & NoSQL',
                description: '데이터의 용도에 맞는 스키마 설계와 비정형 데이터 저장소를 익힙니다.',
                topics: ['Star Schema vs Snowflake Schema', 'Dimensional Modeling (Kimball)', 'MongoDB (Document)', 'Cassandra (Wide-Column)'],
                resources: [
                    { name: 'Data Warehouse Toolkit', url: 'https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/' }
                ],
                quiz: {
                    question: '데이터 웨어하우스 모델링에서 "Fact Table"과 "Dimension Table"로 구성된 가장 일반적인 스키마 형태는?',
                    options: ['Star Schema', 'Relational Schema', 'Graph Schema', 'Network Schema'],
                    correctAnswer: 0
                }
            },
            // 3. Data Warehousing
            {
                step: 'Phase 5: 클라우드 데이터 웨어하우스',
                title: 'Modern Data Warehouse',
                description: '클라우드 환경에서 페타바이트급 데이터를 분석할 수 있는 차세대 DW를 배웁니다.',
                topics: ['Snowflake Architecture', 'Google BigQuery', 'Amazon Redshift', 'Columnar Storage'],
                resources: [
                    { name: 'Snowflake Documentation', url: 'https://docs.snowflake.com/' }
                ],
                quiz: {
                    question: 'Snowflake나 BigQuery 같은 모던 DW가 기존 DB와 달리 대용량 분석에 빠른 이유는 데이터를 어떻게 저장하기 때문인가?',
                    options: ['Columnar Storage (컬럼 기반 저장)', 'Row-based Storage', 'Text Files', 'Linked List'],
                    correctAnswer: 0
                }
            },
            // 4. Data Processing (ETL/ELT)
            {
                step: 'Phase 6: 데이터 파이프라인 (Orchestration)',
                title: 'Workflow Orchestration',
                description: '복잡한 데이터 의존성을 관리하고 자동화하는 워크플로우 도구를 사용합니다.',
                topics: ['Apache Airflow', 'DAGs (Directed Acyclic Graphs)', 'Scheduling & Backfill', 'Task Operators'],
                resources: [
                    { name: 'Apache Airflow Fundamentals', url: 'https://airflow.apache.org/' }
                ],
                quiz: {
                    question: 'Airflow에서 과거의 특정 시점으로 돌아가서 해당 기간의 데이터를 다시 처리하는 작업을 무엇이라 하는가?',
                    options: ['Backfill', 'Restore', 'Rollback', 'Retry'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 7: 데이터 변환 (Transformation)',
                title: 'Transformation with dbt',
                description: 'SQL만으로 데이터 웨어하우스 내부에서 데이터를 변환하고 테스트하는 ELT 방식을 익힘니다.',
                topics: ['dbt (data build tool)', 'Jinja Templating', 'Data Lineage', 'Data Quality Tests'],
                resources: [
                    { name: 'dbt Fundamentals', url: 'https://courses.getdbt.com/courses/fundamentals' }
                ],
                quiz: {
                    question: 'dbt(data build tool)가 채택하고 있는 데이터 처리 방식은?',
                    options: ['ELT (Extract-Load-Transform)', 'ETL (Extract-Transform-Load)', 'Streaming', 'Batch Only'],
                    correctAnswer: 0
                }
            },
            // 5. Big Data & Distributed Computing
            {
                step: 'Phase 8: 빅데이터 프레임워크 (Spark)',
                title: 'Apache Spark',
                description: '인메모리 분산 처리 기술을 이용해 대규모 데이터를 빠르게 처리합니다.',
                topics: ['RDD vs DataFrame', 'SparkSQL', 'Optimization (Catalyst Optimizer)', 'PySpark'],
                resources: [
                    { name: 'Spark The Definitive Guide', url: 'https://github.com/databricks/Spark-The-Definitive-Guide' }
                ],
                quiz: {
                    question: 'Spark에서 작업을 지연 실행(Lazy Evaluation)하며, 실제 결과가 필요할 때(Action) 연산을 수행하는 구조의 장점은?',
                    options: ['실행 계획 최적화 가능', '메모리 사용량 증가', '디버깅이 쉬워짐', '코드가 복잡해짐'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 9: 분산 저장소 (Data Lake)',
                title: 'Data Lake & Formats',
                description: 'Raw 데이터를 저렴하고 안전하게 저장하는 기술과 효율적인 파일 포맷을 다룹니다.',
                topics: ['Hadoop HDFS', 'AWS S3 / GCS', 'Parquet / Avro / ORC', 'Data Partitioning'],
                resources: [
                    { name: 'Parquet File Format', url: 'https://parquet.apache.org/' }
                ],
                quiz: {
                    question: '빅데이터 저장 포맷 중 컬럼 기반이며 압축률이 높아서 분석 용도로 가장 널리 쓰이는 포맷은?',
                    options: ['Parquet', 'CSV', 'JSON', 'XML'],
                    correctAnswer: 0
                }
            },
            // 6. Streaming Data
            {
                step: 'Phase 10: 메세지 큐 & 스트리밍',
                title: 'Message Queues',
                description: '시스템 간의 비동기 데이터 전달을 위한 메시징 미들웨어를 익힙니다.',
                topics: ['Apache Kafka', 'Producer & Consumer', 'Topics & Partitions', 'RabbitMQ'],
                resources: [
                    { name: 'Kafka The Definitive Guide', url: 'https://kafka.apache.org/documentation/' }
                ],
                quiz: {
                    question: 'Kafka에서 데이터를 분산 저장하고 병렬 처리를 가능하게 하는 기본 단위는?',
                    options: ['Partition', 'Topic', 'Broker', 'Zookeeper'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 실시간 스트림 처리',
                title: 'Stream Processing',
                description: '끊임없이 들어오는 데이터를 실시간으로 집계하고 분석합니다.',
                topics: ['Spark Streaming', 'Apache Flink', 'Window Functions (Tumpling/Sliding)', 'State Management'],
                resources: [
                    { name: 'Apache Flink Docs', url: 'https://flink.apache.org/' }
                ],
                quiz: {
                    question: '스트림 처리에서 "이벤트 발생 시간(Event Time)"과 "처리 시간(Processing Time)"의 차이를 보정하기 위해 사용하는 개념은?',
                    options: ['Watermark', 'Timestamp', 'Latency', 'Checkpoint'],
                    correctAnswer: 0
                }
            },
            // 7. Cloud Infrastructure
            {
                step: 'Phase 12: 클라우드 인프라 활용',
                title: 'Cloud Data Platforms',
                description: 'AWS, GCP, Azure의 관리형 데이터 서비스를 조합하여 아키텍처를 구성합니다.',
                topics: ['AWS Glue / EMR', 'GCP Dataflow / Dataproc', 'Azure Synapse', 'Serverless Functions'],
                resources: [
                    { name: 'AWS Data Analytics', url: 'https://aws.amazon.com/big-data/datalakes-and-analytics/' }
                ],
                quiz: {
                    question: 'AWS의 완전 관리형 서버리스 ETL 서비스는?',
                    options: ['AWS Glue', 'EC2', 'RDS', 'S3'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 13: 인프라 자동화 (IaC)',
                title: 'Infrastructure as Code',
                description: '복잡한 클라우드 리소스를 코드로 정의하여 버전 관리하고 배포합니다.',
                topics: ['Terraform', 'Pulumi', 'AWS CloudFormation', 'State Management'],
                resources: [
                    { name: 'Terraform Registry', url: 'https://registry.terraform.io/' }
                ],
                quiz: {
                    question: 'Terraform을 사용하여 여러 클라우드 프로바이더(AWS, GCP 등)의 리소스를 통합 관리할 수 있는가?',
                    options: ['가능하다 (Provider 기반)', '불가능하다', 'AWS만 가능하다', 'GCP만 가능하다'],
                    correctAnswer: 0
                }
            },
            // 8. CI/CD & Containers
            {
                step: 'Phase 14: 컨테이너화',
                title: 'Docker & Kubernetes',
                description: '데이터 애플리케이션의 배포 환경을 표준화하고 격리합니다.',
                topics: ['Dockerizing ETL Scripts', 'Kubernetes for Spark/Airflow', 'Helm Charts', 'Container Registry'],
                resources: [
                    { name: 'Kubernetes for Data Engineering', url: 'https://medium.com/' }
                ],
                quiz: {
                    question: 'Airflow나 Spark 같은 데이터 도구를 Kubernetes 위에서 운영할 때의 장점으로 가장 적절하지 않은 것은?',
                    options: ['항상 고정된 리소스만 사용 가능하다', '확장성(Scalability)이 좋다', '환경 격리가 잘 된다', '배포 관리가 용이하다'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 15: CI/CD for Data',
                title: 'Data Ops',
                description: '구축한 파이프라인 코드의 테스트와 배포를 자동화합니다.',
                topics: ['GitHub Actions', 'Unit Testing for Data Pipelines', 'Data Quality Gates', 'Blue/Green Deployment'],
                resources: [
                    { name: 'The DataOps Manifesto', url: 'https://dataopsmanifesto.org/' }
                ],
                quiz: {
                    question: 'DataOps의 핵심 원칙 중 하나로, 데이터 파이프라인의 변경 사항을 지속적으로 검증하는 것은?',
                    options: ['Continuous Testing', 'Manual Review', 'Once-a-year Release', 'No Documentation'],
                    correctAnswer: 0
                }
            },
            // 9. Governance & Quality
            {
                step: 'Phase 16: 데이터 품질 관리',
                title: 'Data Quality & Testing',
                description: '데이터가 예상대로 들어오는지 검증하고 오류를 방지합니다.',
                topics: ['Great Expectations', 'Soda Core', 'Data Validation Rules', 'Alerting'],
                resources: [
                    { name: 'Great Expectations Docs', url: 'https://docs.greatexpectations.io/' }
                ],
                quiz: {
                    question: '데이터 파이프라인 실행 중에 "NULL 값이 없어야 함", "값의 범위는 0~100 사이" 등의 조건을 검사하는 도구는?',
                    options: ['Great Expectations', 'Great Wall', 'Data Checker', 'Null Buster'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 17: 데이터 거버넌스',
                title: 'Data Governance',
                description: '데이터의 소유권, 보안, 카탈로그, 그리고 흐름(Lineage)을 관리합니다.',
                topics: ['Data Catalog (DataHub/Amundsen)', 'Data Lineage', 'PII Security', 'Access Control (RBAC)'],
                resources: [
                    { name: 'DataHub Project', url: 'https://datahubproject.io/' }
                ],
                quiz: {
                    question: '데이터가 어디서 생성되어 어떻게 변환되고 어디로 흘러가는지를 시각화한 것을 무엇이라 하는가?',
                    options: ['Data Lineage', 'Data Flow Diagram', 'ER Diagram', 'Data Map'],
                    correctAnswer: 0
                }
            },
            // 10. Advanced Architectures
            {
                step: 'Phase 18: 데이터 레이크하우스',
                title: 'Lakehouse Architecture',
                description: '데이터 레이크의 유연성과 웨어하우스의 관리 기능을 결합한 최신 아키텍처입니다.',
                topics: ['Delta Lake', 'Apache Iceberg', 'Apache Hudi', 'Table Formats'],
                resources: [
                    { name: 'Databricks Lakehouse', url: 'https://www.databricks.com/product/data-lakehouse' }
                ],
                quiz: {
                    question: '데이터 레이크 상에서 ACID 트랜잭션, 스키마 관리, 시간 여행(Time Travel) 기능을 제공하는 오픈 테이블 포맷은?',
                    options: ['Apache Iceberg / Delta Lake', 'CSV', 'Parquet', 'JSON'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 19: 데이터 메시 (Data Mesh)',
                title: 'Data Mesh Concept',
                description: '중앙 집중식 관리에서 벗어나 도메인 주도적인 분산 데이터 아키텍처를 이해합니다.',
                topics: ['Domain-Oriented Design', 'Data as a Product', 'Self-serve Infrastructure', 'Federated Governance'],
                resources: [
                    { name: 'Data Mesh by Zhamak Dehghani', url: 'https://martinfowler.com/articles/data-mesh-principles.html' }
                ],
                quiz: {
                    question: 'Data Mesh의 4대 원칙 중 하나로, 데이터를 단순한 자산이 아닌 "제품"처럼 관리하고 서비스해야 한다는 원칙은?',
                    options: ['Data as a Product', 'Data Centralization', 'Data Hoarding', 'Data Silo'],
                    correctAnswer: 0
                }
            },
            // 11. Career
            {
                step: 'Phase 20: 캡스톤 프로젝트',
                title: 'End-to-End Project',
                description: '크롤링부터 대시보드 시각화까지 전 과정을 아우르는 나만의 프로젝트를 만듭니다.',
                topics: ['Real-time Dashboard Project', 'Data Pipeline Project', 'Blog Writing', 'Resume Review'],
                resources: [
                    { name: 'Start Data Engineering', url: 'https://www.startdataengineering.com/' }
                ],
                quiz: {
                    question: '데이터 엔지니어링 프로젝트에서 기술력만큼이나 중요한 것은?',
                    options: ['비즈니스 문제 해결 능력과 문서화', '무조건 최신 툴 사용', '코드 라인 수', '복잡한 알고리즘'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 21: 시스템 디자인 면접',
                title: 'System Design Interview',
                description: '대규모 데이터 시스템 설계 질문에 대비합니다.',
                topics: ['Scalability', 'Reliability', 'Maintainability', 'Back-of-the-envelope Calculation'],
                resources: [
                    { name: 'Designing Data-Intensive Applications', url: 'https://dataintensive.net/' }
                ],
                quiz: {
                    question: '시스템 디자인 인터뷰에서 "Reliability(신뢰성)"의 의미는?',
                    options: ['결함이나 오류가 있어도 시스템이 올바르게 기능을 지속하는 능력', '시스템의 처리 속도', '시스템의 확장 가능성', '시스템의 개발 비용'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '비전공자도 할 수 있나요?', answer: 'DB와 SQL에 대한 이해가 깊다면 충분히 도전할 수 있습니다. 꼼꼼함과 안정성을 추구하는 성향이 잘 맞습니다.' },
            { question: 'AI랑 관련이 있나요?', answer: '모든 AI 모델은 데이터 위에서 만들어집니다. 양질의 데이터를 공급하는 데이터 엔지니어가 없으면 AI 개발도 불가능합니다.' }
        ]
    },
    {
        id: 'data-sci',
        title: 'Data Scientist',
        description: '다양한 데이터에서 머신러닝과 통계 기술로 숨겨진 인사이트를 찾아내고, 비즈니스 의사결정을 돕는 데이터 전략가입니다.',
        long_description: 'Data Scientist는 데이터를 통해 "가치"를 발견하는 탐험가이자 전략가입니다. 비즈니스 문제를 정의하고, 수집된 대규모 데이터에서 패턴을 찾으며(EDA), 통계적 기법과 머신러닝 모델링을 통해 가설을 검증합니다. 단순히 모델을 만드는 것을 넘어, 분석된 결과를 시각화하고 경영진이 이해할 수 있는 인사이트로 변환하여 실질적인 비즈니스 솔루션을 제안합니다.',
        salary_range: '초봉 4,000 ~ 6,000만원',
        difficulty: 'Hard',
        demand: 'High',
        responsibilities: [
            '비즈니스 문제 정의 및 실험 설계 (A/B Test)',
            '대규모 데이터 수집, 전처리 및 탐색적 분석 (EDA)',
            '예측 모델링 및 머신러닝 알고리즘 개발/검증',
            '데이터 시각화 및 비즈니스 전략 제안'
        ],
        tech_stack: [
            { category: 'Language', skills: ['Python', 'R', 'SQL', 'Scala'] },
            { category: 'Analysis', skills: ['Pandas', 'NumPy', 'SciPy', 'Jupyter'] },
            { category: 'ML & Stat', skills: ['Scikit-learn', 'XGBoost', 'TensorFlow', 'PyTorch', 'Statsmodels'] },
            { category: 'Visualization', skills: ['Matplotlib', 'Seaborn', 'Tableau', 'PowerBI', 'Looker'] }
        ],
        tags: ['Analysis', 'Statistics', 'Strategy', 'Math'],
        focus_areas: [
            'Statistical Analysis & Hypothesis Testing',
            'Machine Learning Modeling',
            'Business Insight & Visualization'
        ],
        roadmap: [
            // 1. Math & Statistics
            {
                step: 'Phase 1: 수학적 사고와 통계',
                title: 'Math & Statistics',
                description: '데이터 뒤에 숨은 진실을 파악하기 위한 확률과 통계의 기초를 다집니다.',
                topics: ['Descriptive Statistics', 'Probability Distributions', 'Hypothesis Testing (p-value, t-test)', 'Bayesian Inference'],
                resources: [
                    { name: 'Khan Academy Statistics', url: 'https://www.khanacademy.org/math/statistics-probability' }
                ],
                quiz: {
                    question: '가설 검정에서 "귀무가설(Null Hypothesis)이 참인데도 기각해버릴 확률"을 무엇이라 하는가?',
                    options: ['Type I Error (1종 오류)', 'Type II Error (2종 오류)', 'Confidence Interval', 'Standard Deviation'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 선형대수와 미적분',
                title: 'Linear Algebra & Calculus',
                description: '머신러닝 알고리즘의 작동 원리를 이해하기 위한 수학을 배웁니다.',
                topics: ['Vectors & Matrices', 'Eigenvalues & Eigenvectors', 'Derivatives & Gradients', 'Cost Functions'],
                resources: [
                    { name: '3Blue1Brown Linear Algebra', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab' }
                ],
                quiz: {
                    question: '주성분 분석(PCA)에서 데이터의 분산이 가장 큰 방향을 나타내는 벡터는?',
                    options: ['Eigenvector (고유벡터)', 'Zero Vector', 'Unit Vector', 'Support Vector'],
                    correctAnswer: 0
                }
            },
            // 2. Programming & Data
            {
                step: 'Phase 3: Python 데이터 분석',
                title: 'Python for Data Science',
                description: 'Python 생태계의 분석 도구를 자유자재로 다룹니다.',
                topics: ['Pandas Advanced (MultiIndex, Pivot)', 'NumPy Broadcasting', 'Data Cleaning & Preprocessing', 'Lambda Functions'],
                resources: [
                    { name: 'Python for Data Analysis Book', url: 'https://wesmckinney.com/book/' }
                ],
                quiz: {
                    question: 'Pandas에서 결측치(NaN)를 특정 값으로 채우는 메서드는?',
                    options: ['fillna()', 'dropna()', 'isna()', 'replace()'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 4: SQL과 데이터베이스',
                title: 'SQL & Database',
                description: '원천 데이터를 직접 추출하고 가공하는 쿼리 능력을 기릅니다.',
                topics: ['Complex Joins', 'Window Functions', 'CTEs (Common Table Expressions)', 'NoSQL Basics (MongoDB)'],
                resources: [
                    { name: 'SQLZoo', url: 'https://sqlzoo.net/' }
                ],
                quiz: {
                    question: 'SQL 쿼리 실행 순서(Logical Processing Order) 중 가장 먼저 실행되는 절은?',
                    options: ['FROM', 'SELECT', 'WHERE', 'ORDER BY'],
                    correctAnswer: 0
                }
            },
            // 3. Machine Learning
            {
                step: 'Phase 5: 머신러닝 기초',
                title: 'Machine Learning Basics',
                description: '데이터에서 패턴을 학습하는 고전적인 머신러닝 알고리즘을 익힙니다.',
                topics: ['Linear/Logistic Regression', 'Decision Trees', 'K-Means Clustering', 'Bias-Variance Tradeoff'],
                resources: [
                    { name: 'Scikit-Learn User Guide', url: 'https://scikit-learn.org/stable/user_guide.html' }
                ],
                quiz: {
                    question: '지도 학습(Supervised Learning)에 해당하지 않는 알고리즘은?',
                    options: ['K-Means Clustering', 'Linear Regression', 'Support Vector Machine', 'Random Forest'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 6: 앙상블 기법',
                title: 'Ensemble Methods',
                description: '여러 모델을 조합하여 예측 성능을 극대화하는 기법을 배웁니다.',
                topics: ['Random Forest', 'Gradient Boosting (XGBoost/LightGBM)', 'Stacking & Bagging', 'Hyperparameter Tuning'],
                resources: [
                    { name: 'XGBoost Documentation', url: 'https://xgboost.readthedocs.io/' }
                ],
                quiz: {
                    question: '이전 모델이 틀린 오차(Residual)를 학습하여 성능을 개선해 나가는 앙상블 방식은?',
                    options: ['Boosting', 'Bagging', 'Voting', 'Stacking'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 7: 딥러닝 입문',
                title: 'Deep Learning Intro',
                description: '비정형 데이터(이미지, 텍스트) 처리를 위한 신경망 기초를 다룹니다.',
                topics: ['Neural Networks (MLP)', 'Activation Functions (ReLU, Sigmoid)', 'Overfitting & Regularization', 'TensorFlow/Keras Basics'],
                resources: [
                    { name: 'Deep Learning Specialization', url: 'https://www.coursera.org/specializations/deep-learning' }
                ],
                quiz: {
                    question: '신경망에서 뉴런의 출력을 비선형으로 변환해주는 함수를 무엇이라 하는가?',
                    options: ['Activation Function (활성화 함수)', 'Loss Function', 'Optimizer', 'Kernel'],
                    correctAnswer: 0
                }
            },
            // 4. Data Visualization & Storytelling
            {
                step: 'Phase 8: 데이터 시각화',
                title: 'Data Visualization',
                description: '데이터를 직관적인 그래프로 표현하여 패턴을 발견합니다.',
                topics: ['Matplotlib & Seaborn', 'Interactive Plots (Plotly)', 'Color Theory', 'Chart Selection Guidelines'],
                resources: [
                    { name: 'Data to Viz', url: 'https://www.data-to-viz.com/' }
                ],
                quiz: {
                    question: '시간의 흐름에 따른 데이터의 변화 추세를 보여주기에 가장 적합한 차트는?',
                    options: ['Line Chart', 'Pie Chart', 'Scatter Plot', 'Bar Chart'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 9: 비즈니스 인텔리전스 (BI)',
                title: 'Business Intelligence',
                description: '경영진을 위한 대시보드를 만들고 리포팅합니다.',
                topics: ['Tableau / PowerBI', 'Dashboard Design', 'Data Storytelling', 'KPI Metrics Definition'],
                resources: [
                    { name: 'Storytelling with Data', url: 'https://www.storytellingwithdata.com/' }
                ],
                quiz: {
                    question: '좋은 대시보드의 조건으로 적절하지 않은 것은?',
                    options: ['최대한 많은 정보를 한 화면에 빽빽하게 담는다', '핵심 지표(KPI)가 잘 보이게 배치한다', '사용자(청중)의 수준을 고려한다', '적절한 색상을 사용하여 강조한다'],
                    correctAnswer: 0
                }
            },
            // 5. Advanced Topics
            {
                step: 'Phase 10: 자연어 처리 (NLP)',
                title: 'Natural Language Processing',
                description: '텍스트 데이터를 분석하여 감정, 주제, 의미를 파악합니다.',
                topics: ['Text Preprocessing (Tokenization)', 'TF-IDF & Word Embeddings', 'Sentiment Analysis', 'Basic Transformers'],
                resources: [
                    { name: 'Hugging Face Course', url: 'https://huggingface.co/course/chapter1/1' }
                ],
                quiz: {
                    question: '단어를 벡터 공간에 매핑하여 단어 간의 의미적 유사도를 계산할 수 있게 하는 기술은?',
                    options: ['Word Embedding', 'One-hot Encoding', 'Label Encoding', 'Hashing'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 시계열 분석',
                title: 'Time Series Analysis',
                description: '시간의 흐름에 있는 데이터를 분석하고 미래를 예측합니다.',
                topics: ['ARIMA/SARIMA', 'Prophet', 'Seasonality & Trend', 'LSTM for Time Series'],
                resources: [
                    { name: 'Forecasting: Principles and Practice', url: 'https://otexts.com/fpp3/' }
                ],
                quiz: {
                    question: '시계열 데이터에서 일정한 주기로 반복되는 패턴을 무엇이라 하는가?',
                    options: ['Seasonality (계절성)', 'Trend (추세)', 'Noise (잡음)', 'Stationarity (정상성)'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 12: 추천 시스템',
                title: 'Recommender Systems',
                description: '사용자의 취향을 분석하여 맞춤형 콘텐츠를 제안하는 알고리즘을 배웁니다.',
                topics: ['Collaborative Filtering', 'Content-based Filtering', 'Matrix Factorization', 'Hybrid Models'],
                resources: [
                    { name: 'Google Recommendation Systems', url: 'https://developers.google.com/machine-learning/recommendation' }
                ],
                quiz: {
                    question: '"나와 비슷한 취향을 가진 다른 사용자가 구매한 상품"을 추천해주는 방식은?',
                    options: ['User-based Collaborative Filtering', 'Content-based Filtering', 'Random Recommendation', 'Popularity-based'],
                    correctAnswer: 0
                }
            },
            // 6. Big Data Ecosystem
            {
                step: 'Phase 13: 빅데이터 처리',
                title: 'Big Data Frameworks',
                description: '로컬 머신에서 처리할 수 없는 대용량 데이터를 다룹니다.',
                topics: ['Apache Spark (PySpark)', 'Hadoop Ecosystem (Hive)', 'Distributed Computing Concept', 'Data Lakes'],
                resources: [
                    { name: 'PySpark Documentation', url: 'https://spark.apache.org/docs/latest/api/python/' }
                ],
                quiz: {
                    question: 'Spark에서 데이터를 처리하는 기본 추상화 객체로, 분산 데이터 컬렉션을 의미하는 것은?',
                    options: ['RDD (Resilient Distributed Dataset)', 'Pandas DataFrame', 'List', 'Array'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 14: 클라우드 컴퓨팅',
                title: 'Cloud for Data Science',
                description: 'AWS, GCP 등 클라우드 환경에서 분석 환경을 구축합니다.',
                topics: ['AWS SageMaker', 'Google Vertex AI', 'Docker Basics', 'Serverless Functions'],
                resources: [
                    { name: 'AWS Setup for Data Science', url: 'https://aws.amazon.com/training/' }
                ],
                quiz: {
                    question: 'AWS에서 Jupyter Notebook 환경을 완벽하게 관리형으로 제공하는 머신러닝 서비스는?',
                    options: ['SageMaker', 'EC2', 'S3', 'Lambda'],
                    correctAnswer: 0
                }
            },
            // 7. Experimentation & Research
            {
                step: 'Phase 15: 실험 설계 (A/B Testing)',
                title: 'A/B Testing',
                description: '비즈니스 변경 사항의 효과를 통계적으로 검증합니다.',
                topics: ['Control vs Treatment Group', 'Sample Size Calculation', 'Statistical Significance', 'Metric Selection'],
                resources: [
                    { name: 'A/B Testing Guide', url: 'https://vwo.com/ab-testing/' }
                ],
                quiz: {
                    question: 'A/B 테스트에서 실험 결과를 신뢰할 수 있는지 판단하기 위해 확인하는 통계적 지표는?',
                    options: ['p-value', 'Accuracy', 'Loss', 'Epoch'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 16: 인과 추론',
                title: 'Causal Inference',
                description: '상관관계를 넘어 진짜 원인과 결과를 파악합니다.',
                topics: ['Correlation vs Causation', 'Simpson\'s Paradox', 'Propensity Score Matching', 'Diverse Counterfactuals'],
                resources: [
                    { name: 'The Book of Why', url: 'http://bayes.cs.ucla.edu/WHY/' }
                ],
                quiz: {
                    question: '"아이스크림 판매량과 익사 사고 건수는 양의 상관관계가 있다"는 명제에서 숨겨진 제3의 변수(Confounder)는?',
                    options: ['기온 (여름 날씨)', '상어의 수', '아이스크림 가격', '수영장 크기'],
                    correctAnswer: 0
                }
            },
            // 8. Career & Ethics
            {
                step: 'Phase 17: 데이터 윤리',
                title: 'Data Ethics & Privacy',
                description: '데이터 분석가로서 지켜야 할 윤리적 책임과 규제를 이해합니다.',
                topics: ['GDPR/CCPA', 'Bias in AI', 'Data Anonymization', 'Responsible AI'],
                resources: [
                    { name: 'Data Ethics Course', url: 'https://ethics.fast.ai/' }
                ],
                quiz: {
                    question: '학습 데이터의 편향(Bias)으로 인해 모델이 특정 인구 집단에 불리한 예측을 하는 현상은?',
                    options: ['Algorithmic Bias', 'Overfitting', 'Underfitting', 'Feature Selection'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 18: 포트폴리오 프로젝트',
                title: 'Capstone Project',
                description: '자신만의 가설을 세우고 데이터를 수집/분석하여 결론을 도출하는 전체 과정을 수행합니다.',
                topics: ['Project Scoping', 'Data Collection (Crawling)', 'Modeling', 'Storytelling/Reporting'],
                resources: [
                    { name: 'Kaggle Datasets', url: 'https://www.kaggle.com/datasets' }
                ],
                quiz: {
                    question: '데이터 사이언스 포트폴리오에서 가장 중요한 요소는?',
                    options: ['어떤 문제를 해결했고 어떤 비즈니스 임팩트를 냈는지 설명하는 능력', '가장 복잡한 딥러닝 모델 사용', '코드의 줄 수', '데이터의 크기'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 19: 이력서 및 면접 준비',
                title: 'Career Preparation',
                description: '데이터 사이언티스트 면접을 통과하기 위한 실전 팁을 익힙니다.',
                topics: ['Technical Interview', 'Behavioral Interview', 'Resume Writing', 'Take-home Assignment'],
                resources: [
                    { name: 'Data Science Interview Prep', url: 'https://www.interviewquery.com/' }
                ],
                quiz: {
                    question: '면접관이 "프로젝트에서 가장 어려웠던 점은 무엇인가요?"라고 물었을 때 가장 좋은 답변 방식은?',
                    options: ['문제 상황(S) -> 행동(A) -> 결과(R) 구조로 구체적으로 설명', '어려운 점이 없었다고 답변', '팀원 탓을 한다', '기억이 안 난다고 한다'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 20: 최신 트렌드 팔로우',
                title: 'Continuing Education',
                description: 'LLM, GenAI 등 끊임없이 변화하는 데이터 분야의 최신 기술을 학습합니다.',
                topics: ['LLM Applications', 'RAG (Retrieval-Augmented Generation)', 'Prompt Engineering', 'AI Newsletters'],
                resources: [
                    { name: 'The Batch (DeepLearning.AI)', url: 'https://www.deeplearning.ai/the-batch/' }
                ],
                quiz: {
                    question: '최근 데이터 사이언스 분야에서 가장 뜨거운 화두인 "생성형 AI"의 영어 약자는?',
                    options: ['GenAI', 'AGI', 'XAI', 'AutoML'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 21: 커리어 성장전략',
                title: 'Career Growth',
                description: '주니어에서 시니어로, 또는 매니저로 성장하기 위한 로드맵을 그립니다.',
                topics: ['Mentoring', 'Networking', 'Specialization (NLP/CV/Inference)', 'Leadership'],
                resources: [
                    { name: 'Staff Engineer Path', url: 'https://staffeng.com/' }
                ],
                quiz: {
                    question: '개인 기여(IC) 단계를 넘어 팀 전체의 기술적 의사결정을 주도하는 엔지니어 직군은?',
                    options: ['Staff/Principal Engineer', 'Junior Engineer', 'Intern', 'Contractor'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '석/박사 학위가 필수인가요?', answer: '필수는 아니지만, 통계적 지식의 깊이가 요구되므로 우대하는 경향이 있습니다. 포트폴리오로 실력을 증명하면 학력은 극복 가능합니다.' },
            { question: '취업이 어렵나요?', answer: '단순 분석가는 포화 상태지만, 비즈니스 감각과 엔지니어링 능력을 겸비한 "Full-stack Data Scientist"는 여전히 수요가 많습니다.' }
        ]
    },
    {
        id: 'research',
        title: 'AI Research Engineer',
        description: '최신 AI 논문의 아이디어를 가장 빠르고 효율적인 코드로 현실화하여 시스템에 적용하는 연구 개발 전문가입니다.',
        long_description: 'AI Research Engineer는 "연구(Research)"와 "구현(Engineering)"의 가교 역할을 합니다. Research Scientist가 제안한 혁신적인 알고리즘 가설을 실제 작동하는 고성능 시스템으로 변환하고, 수백 대의 GPU를 활용한 대규모 실험이 가능하도록 인프라와 파이프라인을 최적화합니다. 논문의 수식을 코드로 옮기는 능력뿐만 아니라, 모델 경량화와 속도 최적화까지 책임지는 "해결사"입니다.',
        salary_range: '초봉 5,000 ~ 8,000만원 + @',
        difficulty: 'Extreme',
        demand: 'Very High',
        responsibilities: [
            '최신 논문 구현 및 검증 (Paper to Code)',
            '실험 자동화 및 파이프라인 효율화 (MLOps for Research)',
            '대규모 분산 학습 최적화 (Multi-GPU/Node)',
            '모델 경량화 및 추론 속도 개선 (Optimization)'
        ],
        tech_stack: [
            { category: 'Framework', skills: ['PyTorch', 'JAX', 'TensorFlow', 'Hugging Face'] },
            { category: 'Language', skills: ['Python', 'C++', 'CUDA'] },
            { category: 'HPC & Infra', skills: ['Docker', 'Kubernetes', 'DeepSpeed', 'Slurm'] },
            { category: 'Math', skills: ['Linear Algebra', 'Probability', 'Optimization Theory'] }
        ],
        tags: ['DeepLearning', 'HPC', 'PaperImplementation', 'R&D'],
        focus_areas: [
            'Deep Learning Systems (Distributed Training)',
            'Model Optimization (Quantization/Pruning)',
            'Latest Architecture Implementation (LLM/Diffusion)'
        ],
        roadmap: [
            // 1. Foundations
            {
                step: 'Phase 1: 수학적 기초 (Mathematics)',
                title: 'Math for AI',
                description: '논문의 수식을 코드로 옮기기 위한 필수 수학을 다집니다.',
                topics: ['Linear Algebra (Matrix Operations)', 'Calculus (Gradient, Chain Rule)', 'Probability & Statistics', 'Optimization Methods (Adam, L-BFGS)'],
                resources: [
                    { name: 'Mathematics for Machine Learning', url: 'https://mml-book.github.io/' }
                ],
                quiz: {
                    question: '딥러닝 최적화에서 Loss 함수의 최솟값을 찾기 위해 기울기(Gradient)의 반대 방향으로 이동하는 기본적인 방법은?',
                    options: ['Gradient Descent (경사 하강법)', 'Newton Method', 'Genetic Algorithm', 'Random Search'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: CS & Python 심화',
                title: 'CS & Python Expert',
                description: '고성능 연산을 위한 Python 심화 기법과 알고리즘 효율성을 학습합니다.',
                topics: ['Python Decorators & Generators', 'NumPy Broadcasting', 'C++ Basics for AI', 'Time Complexity (Big O)'],
                resources: [
                    { name: 'Grokking Algorithms', url: 'https://www.manning.com/books/grokking-algorithms' }
                ],
                quiz: {
                    question: 'NumPy에서 모양(Shape)이 다른 배열 간의 연산을 가능하게 해주는 메커니즘은?',
                    options: ['Broadcasting', 'Reshaping', 'Flattening', 'Slicing'],
                    correctAnswer: 0
                }
            },
            // 2. Core Deep Learning
            {
                step: 'Phase 3: 딥러닝 프레임워크 (PyTorch)',
                title: 'PyTorch Mastery',
                description: '연구 표준 프레임워크인 PyTorch를 바닥부터 자유자재로 다룹니다.',
                topics: ['Tensor Operations & Autograd', 'Custom Dataset/DataLoader', 'nn.Module Customization', 'Einsum Operations'],
                resources: [
                    { name: 'PyTorch Official Tutorials', url: 'https://pytorch.org/tutorials/' }
                ],
                quiz: {
                    question: 'PyTorch에서 텐서의 연산 기록을 추적하여 자동으로 기울기를 계산해주는 엔진은?',
                    options: ['Autograd', 'AutoML', 'Backprop Engine', 'Gradient Booster'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 4: 논문 구현 기초',
                title: 'Paper Implementation',
                description: '간단한 논문부터 시작해 수식을 코드로 변환하는 훈련을 합니다.',
                topics: ['Reading Papers (Abstract to Conclusion)', 'Code Structure for Research', 'Debuging ML Models', 'Reproducibility'],
                resources: [
                    { name: 'Papers with Code', url: 'https://paperswithcode.com/' }
                ],
                quiz: {
                    question: '논문 구현 시, 실험 결과의 재현성(Reproducibility)을 보장하기 위해 가장 먼저 고정해야 하는 것은?',
                    options: ['Random Seed', 'Learning Rate', 'Batch Size', 'GPU Model'],
                    correctAnswer: 0
                }
            },
            // 3. Architectures
            {
                step: 'Phase 5: CNN & Vision Architectures',
                title: 'Computer Vision',
                description: '이미지 처리를 위한 최신 컨볼루션 아키텍처를 구현합니다.',
                topics: ['ResNet (Skip Connection)', 'EfficientNet (Scaling)', 'Vision Transformer (ViT)', 'Object Detection (YOLO)'],
                resources: [
                    { name: 'CS231n', url: 'http://cs231n.stanford.edu/' }
                ],
                quiz: {
                    question: 'ResNet에서 깊은 신경망의 학습을 가능하게 만든 핵심 구조인 "Skip Connection"이 해결한 문제는?',
                    options: ['Vanishing Gradient (기울기 소실)', 'Overfitting (과적합)', 'Memory Leak', 'Slow Inference'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 6: RNN & Transformers',
                title: 'NLP Architectures',
                description: '시퀀스 데이터를 처리하는 모델의 발전 과정을 따라 구현합니다.',
                topics: ['LSTM/GRU', 'Attention Mechanism', 'Transformer (Self-Attention)', 'BERT & GPT Basics'],
                resources: [
                    { name: 'The Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/' }
                ],
                quiz: {
                    question: 'Transformer 모델의 핵심 메커니즘으로, 입력 시퀀스의 모든 위치 간의 관계를 계산하는 것은?',
                    options: ['Self-Attention', 'Convolution', 'Recurrence', 'Pooling'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 7: 생성형 AI (Generative Models)',
                title: 'Generative AI',
                description: '데이터의 분포를 학습하여 새로운 데이터를 생성하는 모델을 다룹니다.',
                topics: ['VAE (Variational AutoEncoder)', 'GANs (Adversarial Training)', 'Diffusion Models (DDPM)', 'Latent Space'],
                resources: [
                    { name: 'OpenAI Spinning Up', url: 'https://spinningup.openai.com/' }
                ],
                quiz: {
                    question: 'Diffusion Model이 데이터를 생성하는 방식은?',
                    options: ['노이즈를 단계적으로 제거(Denoising)하여 원본 복원', '노이즈를 한 번에 제거', '이미지를 회전시켜 생성', '이미지를 캡션으로 변환'],
                    correctAnswer: 0
                }
            },
            // 4. Optimization & HPC
            {
                step: 'Phase 8: 모델 학습 최적화',
                title: 'Training Optimization',
                description: '학습 속도를 높이고 수렴을 돕는 다양한 기법을 적용합니다.',
                topics: ['Learning Rate Schedulers', 'Weight Initialization', 'Normalization (Batch/Layer/Group)', 'Mixed Precision (FP16/BF16)'],
                resources: [
                    { name: 'NVIDIA Mixed Precision Training', url: 'https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html' }
                ],
                quiz: {
                    question: 'Mixed Precision Training(AMP)을 사용할 때 얻을 수 있는 주요 이점 두 가지는?',
                    options: ['메모리 사용량 감소 및 학습 속도 향상', '모델 정확도 향상 및 오버피팅 방지', '코드 복잡도 감소 및 버그 감소', '데이터 전처리 속도 향상'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 9: 분산 학습 (Distributed Training)',
                title: 'Multi-GPU Training',
                description: '단일 GPU의 한계를 넘어 수백 대의 GPU로 학습하는 기술을 익힙니다.',
                topics: ['Data Parallelism (DDP)', 'Model Parallelism', 'FSDP (Fully Sharded Data Parallel)', 'DeepSpeed / Megatron-LM'],
                resources: [
                    { name: 'PyTorch Distributed Tutorial', url: 'https://pytorch.org/tutorials/beginner/dist_overview.html' }
                ],
                quiz: {
                    question: 'PyTorch에서 가장 권장되는 멀티 GPU 데이터 병렬 처리 방식은?',
                    options: ['DistributedDataParallel (DDP)', 'DataParallel (DP)', 'SingleGPU', 'CPU Parallel'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 10: 고성능 컴퓨팅 (HPC)',
                title: 'HPC & Infrastructure',
                description: '대규모 실험을 위한 하드웨어와 클러스터 환경을 이해합니다.',
                topics: ['CUDA Programming Basics', 'GPU Memory Hierarchy', 'Slurm Workload Manager', 'Introduction to Supercomputing'],
                resources: [
                    { name: 'NVIDIA CUDA Guide', url: 'https://docs.nvidia.com/cuda/' }
                ],
                quiz: {
                    question: 'GPU 프로그래밍에서 CPU를 Host라 부르고, GPU를 무엇이라 부르는가?',
                    options: ['Device', 'Server', 'Client', 'Node'],
                    correctAnswer: 0
                }
            },
            // 5. Research Engineering
            {
                step: 'Phase 11: 실험 관리 (Experiment Tracking)',
                title: 'MLOps for Research',
                description: '수많은 실험의 하이퍼파라미터와 결과를 체계적으로 기록하고 비교합니다.',
                topics: ['Weights & Biases (W&B)', 'MLflow', 'Hydra / OmegaConf (Config Management)', 'Logging Best Practices'],
                resources: [
                    { name: 'Weights & Biases Logic', url: 'https://wandb.ai/' }
                ],
                quiz: {
                    question: '딥러닝 실험에서 Config(설정값)와 결과를 매핑하여 시각화해주는 도구로 가장 널리 쓰이는 것은?',
                    options: ['Weights & Biases', 'Excel', 'Notepad', 'PowerPoint'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 12: 모델 경량화 (Compression)',
                title: 'Model Compression',
                description: '거대 모델을 작은 디바이스에서도 돌릴 수 있게 최적화합니다.',
                topics: ['Quantization (PTQ/QAT)', 'Pruning (Structured/Unstructured)', 'Knowledge Distillation', 'ONNX / TensorRT'],
                resources: [
                    { name: 'TinyML Book', url: 'https://www.oreilly.com/library/view/tinyml/9781492052036/' }
                ],
                quiz: {
                    question: '큰 모델(Teacher)의 지식을 작은 모델(Student)에게 전달하여 학습시키는 기법은?',
                    options: ['Knowledge Distillation', 'Knowledge Transfer', 'Model Resizing', 'Data Augmentation'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 13: 최신 LLM 엔지니어링',
                title: 'LLM Engineering',
                description: 'GPT와 같은 거대 언어 모델을 다루는 기술을 심도 있게 배웁니다.',
                topics: ['PEFT (LoRA, QLoRA)', 'RLHF (PPO/DPO)', 'Prompt Engineering', 'LangChain Basics'],
                resources: [
                    { name: 'Hugging Face PEFT', url: 'https://huggingface.co/docs/peft/index' }
                ],
                quiz: {
                    question: '거대 언어 모델의 모든 파라미터를 튜닝하지 않고, 일부 파라미터만 효율적으로 튜닝하는 기법을 통칭하는 말은?',
                    options: ['PEFT (Parameter-Efficient Fine-Tuning)', 'Full Fine-Tuning', 'Pre-training', 'Zero-shot Learning'],
                    correctAnswer: 0
                }
            },
            // 6. Project & Career
            {
                step: 'Phase 14: 논문 리딩 및 리뷰',
                title: 'Keeping up with AI',
                description: '쏟아지는 ArXiv 논문 홍수 속에서 중요한 연구를 선별하고 빠르게 습득합니다.',
                topics: ['Reading Strategy', 'Twitter/X AI Community', 'Conference (NeurIPS/ICML/CVPR)', 'Writing Review Articles'],
                resources: [
                    { name: 'AlphaSignal Newsletter', url: 'https://alphasignal.ai/' }
                ],
                quiz: {
                    question: 'AI 분야의 Top-tier 학회가 아닌 것은?',
                    options: ['CES', 'NeurIPS', 'ICML', 'ICLR'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 15: 오픈 소스 기여',
                title: 'Open Source Contribution',
                description: 'Hugging Face나 PyTorch 생태계에 기여하여 영향력을 넓힙니다.',
                topics: ['GitHub PR Process', 'Documentation Contribution', 'Model Hub Upload', 'Creating Demo (Gradio/Streamlit)'],
                resources: [
                    { name: 'Hugging Face Transformers Code', url: 'https://github.com/huggingface/transformers' }
                ],
                quiz: {
                    question: 'Hugging Face에 자신의 모델을 업로드하여 다른 사람들과 공유할 수 있는 플랫폼은?',
                    options: ['Hugging Face Model Hub', 'GitHub Gist', 'Docker Hub', 'NPM'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 16: 고급 연구 주제',
                title: 'Advanced Research Topics',
                description: 'AI 연구의 최전선에 있는 주제들을 탐구합니다.',
                topics: ['Meta-Learning (Learning to Learn)', 'Self-Supervised Learning (SimCLR/MAE)', 'Graph Neural Networks (GNN)', 'Neural Architecture Search (NAS)'],
                resources: [
                    { name: 'Lil\'Log SSL', url: 'https://lilianweng.github.io/posts/2021-05-31-contrastive/' }
                ],
                quiz: {
                    question: '레이블이 없는 막대한 데이터로부터 스스로 표현(Representation)을 학습하는 방법론은?',
                    options: ['Self-Supervised Learning', 'Supervised Learning', 'Reinforcement Learning', 'Active Learning'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 17: 강화학습 (Reinforcement Learning)',
                title: 'Deep RL',
                description: '에이전트가 환경과 상호작용하며 보상을 최대화하는 방법을 배웁니다.',
                topics: ['Markov Decision Process (MDP)', 'Q-Learning & DQN', 'Policy Gradient (PPO)', 'AlphaGo Architecture'],
                resources: [
                    { name: 'Deep RL Course (Hugging Face)', url: 'https://huggingface.co/learn/deep-rl-course/unit0/introduction' }
                ],
                quiz: {
                    question: '강화학습에서 에이전트가 취한 행동(Action)에 대해 환경(Environment)이 돌려주는 피드백은?',
                    options: ['Reward (보상)', 'Loss (손실)', 'Gradient (기울기)', 'Label (정답)'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 18: 멀티모달 (Multimodal)',
                title: 'Multimodal AI',
                description: '텍스트, 이미지, 오디오 등 여러 모달리티를 결합하여 이해하는 모델을 다룹니다.',
                topics: ['CLIP (Contrastive Language-Image Pretraining)', 'Stable Diffusion', 'Audio Spectrogram', 'Video Understanding'],
                resources: [
                    { name: 'OpenAI CLIP Paper', url: 'https://openai.com/research/clip' }
                ],
                quiz: {
                    question: 'OpenAI의 CLIP 모델이 학습된 방식은?',
                    options: ['이미지와 텍스트 캡션 쌍의 대조 학습 (Contrastive Learning)', '이미지 분류 학습', '텍스트 생성 학습', '강화 학습'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 19: 캡스톤 프로젝트',
                title: 'SoTA Implementation',
                description: '최신 논문(State-of-the-Art)을 직접 선정하여 바닥부터 구현하고 블로그에 정리합니다.',
                topics: ['Paper Selection', 'Model Arch Implementation', 'Training Loop & Debugging', 'Result Visualization & Blog'],
                resources: [
                    { name: 'Papers with Code', url: 'https://paperswithcode.com/' }
                ],
                quiz: {
                    question: '논문 구현 프로젝트에서 가장 중요한 마음가짐은?',
                    options: ['끈기있게 디버깅하고 원본 저자의 코드를 참고하며 배우는 자세', '무조건 한 번에 성공하기', '모든 코드를 암기하기', '가장 복잡한 모델 고르기'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 20: 리서치 인터뷰 준비',
                title: 'Interview Prep',
                description: 'AI 리서치 엔지니어 면접에서 자주 나오는 질문을 대비합니다.',
                topics: ['Live Coding (Python/PyTorch)', 'Paper Review Presentation', 'Core ML Theory Q&A', 'System Design for ML'],
                resources: [
                    { name: 'ML System Design Interview', url: 'https://github.com/chiphuyen/ml-interviews' }
                ],
                quiz: {
                    question: '면접에서 "Overfitting이 발생했을 때 해결 방법"으로 적절하지 않은 것은?',
                    options: ['모델의 파라미터 수를 대폭 늘린다', 'Dropout을 적용한다', 'Data Augmentation을 사용한다', 'L1/L2 Regularization을 추가한다'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '석/박사 학위가 없어도 되나요?', answer: 'Research Engineer는 구현 능력이 탁월하다면 학사 출신도 충분히 가능합니다. 다만, 최신 논문을 읽고 이해하는 능력은 필수입니다.' },
            { question: 'Data Scientist와 무엇이 다른가요?', answer: 'Data Scientist가 비즈니스 인사이트와 통계적 분석에 집중한다면, Research Engineer는 딥러닝 모델의 "구현", "최적화", "대규모 학습 시스템"에 집중합니다.' }
        ]
    },
    {
        id: 'pm',
        title: 'AI Product Manager',
        description: 'AI 기술을 적용하여 해결할 수 있는 비즈니스 문제를 정의하고, 제품의 기획부터 출시, 개선까지 이르는 전체 생애주기를 관리합니다.',
        long_description: 'AI Product Manager는 기술과 시장을 연결하는 "지휘자"입니다. 사용자 요구사항을 분석하여 AI가 구현할 기능 명세를 작성하고, 기술적 실현 가능성을 검토하여 제품 로드맵을 수립합니다. 데이터 기반으로 성과 지표(KPI)를 설정하고, 개발팀/디자인팀과 협업하여 AI 모델이 실제 비즈니스 가치를 창출하도록 리드합니다.',
        salary_range: '초봉 3,500 ~ 5,500만원',
        difficulty: 'Medium',
        demand: 'High',
        responsibilities: [
            'AI Opportunity Discovery (전략적 기회 발굴)',
            'Product Requirement Definition (PRD 작성 및 유스케이스 정의)',
            'Data-driven Decision Making (A/B 테스트 및 지표 분석)',
            'Stakeholder Management (개발/비즈니스 팀 조율)',
            'AI Ethics & Compliance (윤리적 리스크 관리)'
        ],
        tech_stack: [
            { category: 'Planning', skills: ['Figma', 'Jira', 'Confluence', 'Miro'] },
            { category: 'Data Analysis', skills: ['SQL', 'Tableau', 'Amplitude', 'Excel'] },
            { category: 'AI Literacy', skills: ['Prompt Engineering', 'Model Evaluation Metrics', 'RAG 이해'] }
        ],
        tags: ['Business', 'Strategy', 'Communication', 'Data'],
        focus_areas: [
            'Product Strategy & Roadmap',
            'Data Analytics (SQL/Metrics)',
            'AI Technology Understanding'
        ],
        roadmap: [
            // 1. Fundamentals
            {
                step: 'Phase 1: PM의 역할과 마인드셋',
                title: 'PM Fundamentals',
                description: 'Product Manager의 핵심 역할과 제품 생애주기(PLC)를 이해합니다.',
                topics: ['What is PM?', 'Product Lifecycle', 'Agile & Scrum Basics', 'Design Thinking'],
                resources: [
                    { name: 'Inspired (Book)', url: 'https://www.svpg.com/books/inspired/' }
                ],
                quiz: {
                    question: '제품 개발 방법론 중, 짧은 주기(Sprint)로 개발과 피드백을 반복하며 유연하게 대응하는 방식은?',
                    options: ['Agile (애자일)', 'Waterfall (워터폴)', 'Six Sigma', 'Lean'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 시장 조사와 사용자 이해',
                title: 'Market & User Research',
                description: '고객의 페인 포인트를 찾고 시장의 기회를 분석합니다.',
                topics: ['User Persona', 'User Journey Map', 'Competitor Analysis', 'Tam/Sam/Som'],
                resources: [
                    { name: 'User Interview Guide', url: 'https://www.nngroup.com/articles/user-interviews/' }
                ],
                quiz: {
                    question: '특정 제품이나 서비스를 사용할 대표적인 가상 사용자를 설정하여 니즈와 행동 패턴을 구체화한 것은?',
                    options: ['Persona (페르소나)', 'Stakeholder', 'Focus Group', 'Beta Tester'],
                    correctAnswer: 0
                }
            },
            // 2. Planning & Strategy
            {
                step: 'Phase 3: 문제 정의와 가설 수립',
                title: 'Problem Definition',
                description: '해결해야 할 "진짜 문제"를 정의하고 솔루션 가설을 세웁니다.',
                topics: ['5 Whys', 'Problem Statement', 'Hypothesis Setting', 'MECE Framework'],
                resources: [
                    { name: 'Lean Canvas', url: 'https://leanstack.com/lean-canvas' }
                ],
                quiz: {
                    question: '문제의 근본 원인을 찾기 위해 "왜?"라는 질문을 반복하는 기법은?',
                    options: ['5 Whys', 'SWOT Analysis', 'PEST Analysis', 'Brainstorming'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 4: 제품 전략과 로드맵',
                title: 'Strategy & Roadmapping',
                description: '제품의 비전을 수립하고, 단기/중장기 실행 계획(로드맵)을 그립니다.',
                topics: ['Product Vision & Mission', 'Objective Key Results (OKR)', 'Roadmap Prioritization (RICE/ICE)', 'MVP Strategy'],
                resources: [
                    { name: 'Product Roadmaps Guide', url: 'https://www.aha.io/roadmapping/guide/product-roadmaps' }
                ],
                quiz: {
                    question: '우선순위를 정할 때 도달(Reach), 영향(Impact), 자신감(Confidence), 노력(Effort)을 점수화하는 프레임워크는?',
                    options: ['RICE Score', 'MoSCoW', 'Kano Model', 'Cost-Benefit Analysis'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 5: 요구사항 정의 (PRD)',
                title: 'Writing Requirements',
                description: '개발자와 디자이너가 이해할 수 있는 명확한 제품 요구사항 문서(PRD)를 작성합니다.',
                topics: ['PRD Structure', 'User Stories', 'Acceptance Criteria', 'Functional vs Non-functional'],
                resources: [
                    { name: 'PRD Templates', url: 'https://www.atlassian.com/agile/product-management/requirements' }
                ],
                quiz: {
                    question: '애자일에서 사용자의 관점에서 필요한 기능을 서술하는 형식("As a user, I want to...")은?',
                    options: ['User Story', 'Use Case', 'Flowchart', 'ERD'],
                    correctAnswer: 0
                }
            },
            // 3. AI Literacy
            {
                step: 'Phase 6: AI 기술 이해 (LLM)',
                title: 'AI Literacy: LLMs',
                description: '생성형 AI의 작동 원리와 한계(Hallucination 등)를 이해하여 기획에 반영합니다.',
                topics: ['Transformer Basics', 'Prompt Engineering', 'Temperature & Context Window', 'Fine-tuning vs RAG'],
                resources: [
                    { name: 'Google AI for Everyone', url: 'https://www.coursera.org/learn/ai-for-everyone' }
                ],
                quiz: {
                    question: 'AI가 사실이 아닌 정보를 마치 사실인 것처럼 그럴싸하게 생성해내는 현상은?',
                    options: ['Hallucination (환각)', 'Overfitting', 'Bias', 'Latency'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 7: AI 기술 이해 (Vision/Predictive)',
                title: 'AI Literacy: CV & Predictive',
                description: '이미지 인식, 추천 시스템 등 다양한 AI 기술의 유스케이스를 파악합니다.',
                topics: ['Computer Vision (OCR, Detection)', 'Recommendation Systems', 'Classification vs Regression', 'Clustering'],
                resources: [
                    { name: 'Machine Learning for PMs', url: 'https://bit.ly/3xXyZ' }
                ],
                quiz: {
                    question: '넷플릭스나 유튜브처럼 사용자 취향에 맞는 콘텐츠를 제안하는 AI 시스템은?',
                    options: ['Recommendation System (추천 시스템)', 'Computer Vision', 'Generative AI', 'Voice Recognition'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 8: AI 성능 평가 지표',
                title: 'AI Metrics',
                description: '정확도(Accuracy)뿐만 아니라 F1-Score, Precision, Recall 등 모델 성능 지표를 해석합니다.',
                topics: ['Confusion Matrix', 'Precision/Recall', 'F1-Score', 'BLEU/ROUGE (GenAI Metrics)'],
                resources: [
                    { name: 'Evaluation Metrics for ML', url: 'https://www.evidentlyai.com/' }
                ],
                quiz: {
                    question: '암 환자 예측처럼 "실제 환자를 놓치지 않는 것"이 중요할 때 가장 주의 깊게 봐야 하는 지표는?',
                    options: ['Recall (재현율)', 'Precision (정밀도)', 'Accuracy (정확도)', 'Specificity'],
                    correctAnswer: 0
                }
            },
            // 4. Data & Analytics
            {
                step: 'Phase 9: 데이터 리터러시 & SQL',
                title: 'Data Literacy & SQL',
                description: '직접 데이터를 추출하여 가설을 검증할 수 있는 SQL 능력을 기릅니다.',
                topics: ['Select/From/Where', 'Group By & Aggregation', 'Joins', 'Cohort Analysis'],
                resources: [
                    { name: 'SQL for Data Analysis', url: 'https://mode.com/sql-tutorial/' }
                ],
                quiz: {
                    question: '데이터베이스에서 원하는 조건에 맞는 행(Row)만 필터링하기 위해 사용하는 SQL 구문은?',
                    options: ['WHERE', 'GROUP BY', 'ORDER BY', 'HAVING'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 10: 데이터 시각화',
                title: 'Data Visualization',
                description: 'Tableau나 PowerBI 등을 활용해 데이터를 시각화하고 인사이트를 도출합니다.',
                topics: ['Chart Types', 'Dashboard Design', 'Tableau/PowerBI Basics', 'Storytelling with Data'],
                resources: [
                    { name: 'Storytelling with Data', url: 'https://www.storytellingwithdata.com/' }
                ],
                quiz: {
                    question: '시간의 흐름에 따른 데이터의 추세나 변화를 보여주기에 가장 적합한 차트는?',
                    options: ['Line Chart (선 그래프)', 'Pie Chart', 'Scatter Plot', 'Heatmap'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 실험 설계 (A/B Test)',
                title: 'A/B Testing',
                description: '데이터에 기반하여 두 가지 옵션 중 더 나은 것을 통계적으로 검증합니다.',
                topics: ['Hypothesis Testing', 'Statistical Significance (p-value)', 'Sample Size Calculation', 'Metric Selection'],
                resources: [
                    { name: 'A/B Testing Guide', url: 'https://vwo.com/ab-testing/' }
                ],
                quiz: {
                    question: 'A/B 테스트 결과가 우연에 의한 것이 아니라는 것을 확신할 수 있는 확률적 기준 지표는?',
                    options: ['P-value (유의확률)', 'Conversion Rate', 'Bounce Rate', 'Retention'],
                    correctAnswer: 0
                }
            },
            // 5. Execution & Operations
            {
                step: 'Phase 12: 프로토타이핑 & MVP',
                title: 'Prototyping & MVP',
                description: '최소 기능 제품(MVP)을 빠르게 만들어 시장 반응을 확인합니다.',
                topics: ['Wireframing (Figma)', 'No-code Tools', 'MVP Definition', 'Usability Testing'],
                resources: [
                    { name: 'Figma Tutorials', url: 'https://www.figma.com/resources/learn-design/' }
                ],
                quiz: {
                    question: '고객에게 가치를 제공할 수 있는 최소한의 기능을 갖춘 제품 초기 버전을 무엇이라 하는가?',
                    options: ['MVP (Minimum Viable Product)', 'Prototype', 'Mockup', 'Beta'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 13: 프로젝트 관리 (Delivery)',
                title: 'Project Management',
                description: '일정을 관리하고 리스크를 최소화하며 제품을 출시합니다.',
                topics: ['Jira/Confluence Usage', 'Sprint Planning', 'Risk Management', 'Retrospectives'],
                resources: [
                    { name: 'Atlassian Agile Coach', url: 'https://www.atlassian.com/agile' }
                ],
                quiz: {
                    question: '스프린트가 끝난 후, 팀이 지난 활동을 돌아보며 좋았던 점과 개선할 점을 논의하는 미팅은?',
                    options: ['Retrospective (회고)', 'Daily Standup', 'Sprint Review', 'Backlog Grooming'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 14: AI 제품 운영 (MLOps for PM)',
                title: 'AI Operations',
                description: '출시된 AI 모델의 재학습 주기와 데이터 파이프라인 운영 이슈를 이해합니다.',
                topics: ['Model Drift', 'Monitoring Dashboards', 'Feedback Loops', 'Cost Management'],
                resources: [
                    { name: 'MLOps for Managers', url: 'https://neptune.ai/blog/mlops-for-managers' }
                ],
                quiz: {
                    question: '시간이 지남에 따라 사용자의 행동이나 데이터 패턴이 변하여 모델의 성능이 떨어지는 현상은?',
                    options: ['Data/Concept Drift', 'Bug', 'Server Down', 'Latency'],
                    correctAnswer: 0
                }
            },
            // 6. Ethics & Strategy
            {
                step: 'Phase 15: AI 윤리와 규제',
                title: 'AI Ethics & Law',
                description: 'AI의 편향성, 저작권, 개인정보 보호 등 비즈니스 리스크를 관리합니다.',
                topics: ['Bias & Fairness', 'Privacy (GDPR)', 'Copyright Issues', 'Explainability (XAI)'],
                resources: [
                    { name: 'AI Ethics Guidelines', url: 'https://www.ibm.com/artificial-intelligence/ethics' }
                ],
                quiz: {
                    question: 'AI의 결정 근거를 사람이 이해할 수 있도록 설명 가능하게 만드는 연구 분야는?',
                    options: ['XAI (Explainable AI)', 'Black box AI', 'Deep Learning', 'Reinforcement Learning'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 16: 가격 책정 및 마케팅',
                title: 'Pricing & GTM',
                description: 'AI 제품의 비용 구조(GPU/API 비용)를 고려한 가격 정책과 출시 전략(GTM)을 짭니다.',
                topics: ['Cost Analysis (Token/GPU)', 'Freemium vs Subscription', 'Product Positioning', 'Sales Enablement'],
                resources: [
                    { name: 'SaaS Pricing Strategies', url: 'https://www.priceintelligently.com/blog' }
                ],
                quiz: {
                    question: '제품 출시 시 타겟 고객, 마케팅 채널, 판매 전략 등을 포함하는 종합적인 계획을 무엇이라 하는가?',
                    options: ['Go-To-Market (GTM) Strategy', 'Business Model Canvas', 'Roadmap', 'Blueprint'],
                    correctAnswer: 0
                }
            },
            // 7. Advanced & Career
            {
                step: 'Phase 17: 이해관계자 커뮤니케이션',
                title: 'Stakeholder Management',
                description: '개발자, 디자이너, 경영진 등 다양한 이해관계자와 효과적으로 소통하고 설득합니다.',
                topics: ['Managing Up', 'Conflict Resolution', 'Negotiation', 'Presentation Skills'],
                resources: [
                    { name: 'Crucial Conversations', url: 'https://www.vitalsmarts.com/resource/crucial-conversations/' }
                ],
                quiz: {
                    question: '권한 없이 영향력을 발휘하여 팀을 이끄는 리더십을 표현하는 말은?',
                    options: ['Influence without Authority', 'Micro-management', 'Dictatorship', 'Laissez-faire'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 18: 제품 분석과 성장 (Growth)',
                title: 'Product Growth',
                description: 'AARRR 프레임워크 등을 활용하여 사용자를 유치하고 유지시키는 성장 전략을 실행합니다.',
                topics: ['AARRR Funnel', 'Retention Strategies', 'Viral Loops', 'North Star Metric'],
                resources: [
                    { name: 'Reforge Growth Series', url: 'https://www.reforge.com/' }
                ],
                quiz: {
                    question: '해적 지표(AARRR) 중, 사용자가 제품을 계속 사용하는지(재방문)를 나타내는 단계는?',
                    options: ['Retention (유지)', 'Acquisition (획득)', 'Activatiln (활성화)', 'Revenue (매출)'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 19: 케이스 스터디 (Case Studies)',
                title: 'AI Product Analysis',
                description: '성공한 AI 제품(ChatGPT, Midjourney 등)과 실패 사례를 분석하여 인사이트를 얻습니다.',
                topics: ['Success Factors', 'Failure Analysis', 'Market Fit', 'UX Analysis'],
                resources: [
                    { name: 'Product Hunt', url: 'https://www.producthunt.com/' }
                ],
                quiz: {
                    question: '시장이 원하고 필요로 하는 제품을 만들어 고객을 만족시키는 상태를 뜻하는 용어는?',
                    options: ['Product-Market Fit (PMF)', 'MVP', 'Pivot', 'Unicorn'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 20: 포트폴리오와 면접',
                title: 'Career Prep',
                description: '자신의 경험을 논리적으로 정리한 포트폴리오를 만들고 PM 면접(Product Sense)을 준비합니다.',
                topics: ['Portfolio Building', 'Product Sense Interview', 'Behavioral Questions', 'Mock Interviews'],
                resources: [
                    { name: 'Exponent PM Interview', url: 'https://www.tryexponent.com/' }
                ],
                quiz: {
                    question: '면접에서 상황(Situation), 과제(Task), 행동(Action), 결과(Result) 순서로 경험을 답변하는 기법은?',
                    options: ['STAR Method', 'SWOT', 'PEST', 'MECE'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 21: 미래 기술 트렌드',
                title: 'Future Trends',
                description: '에이전트(Agentic AI), 멀티모달 등 AI의 미래 흐름을 읽고 대비합니다.',
                topics: ['Autonomous Agents', 'Multimodal AI', 'Quantum Computing Impact', 'Web3 & AI'],
                resources: [
                    { name: 'a16z AI Canon', url: 'https://a16z.com/ai-canon/' }
                ],
                quiz: {
                    question: '단순한 질의응답을 넘어, 스스로 계획을 세우고 도구를 사용하여 작업을 수행하는 AI 시스템의 진화 방향은?',
                    options: ['Agentic AI (AI 에이전트)', 'Chatbot', 'Search Engine', 'Plugin'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '개발을 할 줄 알아야 하나요?', answer: '직접 코딩은 안 해도 되지만, 개발자와의 원활한 소통과 기술적 제약사항 이해를 위해 API, DB, AI 모델의 기본 원리는 반드시 알아야 합니다.' },
            { question: '어떤 전공이 유리한가요?', answer: '전공 무관합니다. 인문계열이라도 논리적 사고력, 데이터 분석 능력, 그리고 사용자에 대한 집착이 있다면 훌륭한 PM이 될 수 있습니다.' }
        ]
    },
    {
        id: 'ml-eng',
        title: 'Machine Learning Engineer',
        description: '머신러닝 알고리즘을 활용해 데이터를 학습시키고, 실제 서비스에 적용하여 가치를 창출하는 모델을 개발합니다.',
        long_description: 'Machine Learning Engineer는 데이터를 먹고 자라는 "모델"을 키우는 육성가입니다. 데이터 사이언티스트가 실험적으로 증명한 모델을 실제 서비스에서 쓸 수 있도록 최적화하거나, 적절한 알고리즘을 선택해 비즈니스 문제를 해결하는 모델을 직접 개발합니다. 모델링 능력과 엔지니어링 능력이 모두 필요합니다.',
        salary_range: '초봉 4,000 ~ 6,000만원',
        difficulty: 'Hard',
        demand: 'Very High',
        responsibilities: [
            '머신러닝/딥러닝 모델 설계 및 학습',
            '데이터 전처리 및 피처 엔지니어링 파이프라인 구축',
            '모델 성능 평가 및 하이퍼파라미터 튜닝',
            '모델 경량화 및 추론 최적화'
        ],
        tech_stack: [
            { category: 'Language', skills: ['Python', 'C++'] },
            { category: 'Deep Learning', skills: ['PyTorch', 'TensorFlow', 'Keras'] },
            { category: 'ML Libs', skills: ['Scikit-learn', 'XGBoost', 'LightGBM'] },
            { category: 'Ops', skills: ['Docker', 'FastAPI', 'ONNX'] }
        ],
        tags: ['Modeling', 'Engineering', 'AI'],
        focus_areas: [
            'Deep Learning (PyTorch/TensorFlow)',
            'ML Algorithms & Mathematics',
            'Model Serving & MLOps Basics'
        ],
        roadmap: [
            // 1. Mathematics
            {
                step: 'Phase 1: 수학적 기초 (Mathematics for ML)',
                title: 'Linear Algebra & Calculus',
                description: '머신러닝 알고리즘의 근간이 되는 선형대수와 미적분을 "직관적"으로 이해합니다.',
                topics: ['Vectors & Matrices (행렬 연산)', 'Eigenvalues & Eigenvectors (고유값 분해)', 'Gradient Descent (경사하강법 원리)', 'Partial Derivatives (편미분과 Chain Rule)'],
                resources: [
                    { name: 'Essence of Linear Algebra (3Blue1Brown)', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab' },
                    { name: 'Mathematics for Machine Learning (Book)', url: 'https://mml-book.com/' }
                ],
                quiz: {
                    question: '행렬 A의 고유값(Eigenvalue) λ와 고유벡터(Eigenvector) v의 관계식인 Av = λv에서, 고유벡터가 의미하는 기하학적 의미는?',
                    options: ['회전하지 않고 크기만 변하는 벡터', '가장 길이가 긴 벡터', '원점을 통과하지 않는 벡터', '상수 벡터'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 통계와 확률 (Statistics & Probability)',
                title: 'Probabilistic Thinking',
                description: '불확실성을 다루는 언어인 확률과, 데이터에서 결론을 도출하는 통계를 익힙니다.',
                topics: ['Probability Distributions (가우시안, 베르누이)', 'Bayes Theorem (베이즈 정리)', 'Hypothesis Testing (p-value, t-test)', 'Maximum Likelihood Estimation (MLE)'],
                resources: [
                    { name: 'StatQuest with Josh Starmer', url: 'https://www.youtube.com/user/joshstarmer' }
                ],
                quiz: {
                    question: '베이즈 정리(Bayes Theorem)에서 "사전 확률(Prior)"을 새로운 증거(Evidence)로 업데이트하여 얻는 확률을 무엇이라 하는가?',
                    options: ['사후 확률 (Posterior)', '우도 (Likelihood)', '주변 확률 (Marginal)', '조건부 확률 (Conditional)'],
                    correctAnswer: 0
                }
            },
            // 2. Programming & Tools
            {
                step: 'Phase 3: Python 프로그래밍 심화',
                title: 'Advanced Python for ML',
                description: '단순 문법을 넘어, 효율적인 데이터 처리를 위한 고급 Python 기법을 마스터합니다.',
                topics: ['List Comprehension & Generators', 'Decorators & Context Managers', 'Multiprocessing vs Threading', 'Type Hinting (mypy)'],
                resources: [
                    { name: 'Real Python', url: 'https://realpython.com/' },
                    { name: 'Effective Python (Book)', url: 'https://effectivepython.com/' }
                ],
                quiz: {
                    question: 'Python에서 대용량 데이터를 메모리에 한꺼번에 올리지 않고, 필요할 때마다 하나씩 생성하여 처리하기 위해 사용하는 객체는?',
                    options: ['Generator', 'List', 'Dictionary', 'Set'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 4: 데이터 핸들링 (NumPy & Pandas)',
                title: 'Data Manipulation Expert',
                description: '수치 연산과 정형 데이터 처리를 자유자재로 다룰 수 있어야 합니다.',
                topics: ['Broadcasting & Vectorization (NumPy)', 'Pandas Optimization (Vectorized Operations)', 'Handling Missing Data', 'Time Series Data Handling'],
                resources: [
                    { name: 'Pandas User Guide', url: 'https://pandas.pydata.org/docs/user_guide/index.html' },
                    { name: '100 Numpy Exercises', url: 'https://github.com/rougier/numpy-100' }
                ],
                quiz: {
                    question: 'Pandas에서 반복문(for-loop)을 사용하는 것보다 훨씬 빠르며, 배열 단위로 연산을 수행하는 기법을 무엇이라 하는가?',
                    options: ['Vectorization (벡터화)', 'Serialization (직렬화)', 'Loop Unrolling', 'Memoization'],
                    correctAnswer: 0
                }
            },
            // 3. Data Engineering Basics
            {
                step: 'Phase 5: 데이터베이스와 SQL',
                title: 'SQL & Database for ML',
                description: '모델 학습에 필요한 데이터를 DB에서 직접 추출하고 가공할 줄 알아야 합니다.',
                topics: ['Complex Joins & Subqueries', 'Window Functions (RANK, LEAD/LAG)', 'Indexing & Query Optimization', 'NoSQL Basics (MongoDB, Redis)'],
                resources: [
                    { name: 'Mode SQL Tutorial', url: 'https://mode.com/sql-tutorial/' }
                ],
                quiz: {
                    question: 'SQL에서 그룹별 순위를 매기거나 이동 평균을 구할 때 사용하는 함수 통칭은?',
                    options: ['Window Functions', 'Aggregate Functions', 'Scalar Functions', 'Trigger Functions'],
                    correctAnswer: 0
                }
            },
            // 4. Classical Machine Learning
            {
                step: 'Phase 6: 머신러닝 기초 (Supervised)',
                title: 'Classical Supervised Learning',
                description: '가장 기본이 되는 회귀와 분류 알고리즘의 작동 원리를 바닥부터 구현해봅니다.',
                topics: ['Linear/Logistic Regression', 'Decision Trees', 'Support Vector Machines (SVM)', 'K-Nearest Neighbors (KNN)'],
                resources: [
                    { name: 'Scikit-Learn Documentation', url: 'https://scikit-learn.org/stable/' }
                ],
                quiz: {
                    question: 'SVM(Support Vector Machine)에서 데이터를 고차원으로 매핑하여 비선형 분리를 가능하게 하는 기법은?',
                    options: ['Kernel Trick', 'Regularization', 'Gradient Boosting', 'Feature Selection'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 7: 앙상블 학습 (Ensemble Methods)',
                title: 'Ensemble & Boosting',
                description: '여러 모델을 조합하여 강력한 성능을 내는 배깅과 부스팅 기법을 익힙니다.',
                topics: ['Random Forest (Bagging)', 'XGBoost/LightGBM/CatBoost (Boosting)', 'Stacking & Blending', 'Feature Importance Analysis'],
                resources: [
                    { name: 'XGBoost Documentation', url: 'https://xgboost.readthedocs.io/' }
                ],
                quiz: {
                    question: '이전 모델이 틀린 데이터(오답)에 가중치를 부여하여 다음 모델을 학습시키는 방식의 앙상블 기법은?',
                    options: ['Boosting', 'Bagging', 'Stacking', 'Voting'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 8: 비지도 학습 & 차원 축소',
                title: 'Unsupervised Learning',
                description: '레이블이 없는 데이터에서 패턴을 찾고, 고차원 데이터를 시각화 가능한 형태로 줄입니다.',
                topics: ['K-Means & DBSCAN Clustering', 'PCA (Principal Component Analysis)', 't-SNE & UMAP (Visualization)', 'Anomaly Detection (이상 탐지)'],
                resources: [
                    { name: 'Understanding PCA', url: 'https://setosa.io/ev/principal-component-analysis/' }
                ],
                quiz: {
                    question: '데이터의 분산(Variance)을 최대한 보존하는 축을 찾아 데이터를 투영하여 차원을 축소하는 기법은?',
                    options: ['PCA', 'K-Means', 'Logistic Regression', 'Dropout'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 9: 모델 평가와 실험 설계',
                title: 'Evaluation Metrics & Validation',
                description: '정확도(Accuracy)의 함정을 피하고, 모델의 진짜 성능을 검증하는 방법을 배웁니다.',
                topics: ['Confusion Matrix', 'Precision, Recall, F1-Score', 'ROC Curve & AUC', 'Cross-Validation (K-Fold, Stratified)'],
                resources: [
                    { name: 'Google ML Crash Course - Classification', url: 'https://developers.google.com/machine-learning/crash-course/classification/video-lecture' }
                ],
                quiz: {
                    question: '암 환자 예측 모델에서, 실제 암 환자(Positive)를 정상(Negative)으로 잘못 예측하는 것은 어떤 오류인가?',
                    options: ['Type II Error (False Negative)', 'Type I Error (False Positive)', 'Overfitting', 'Underfitting'],
                    correctAnswer: 0
                }
            },
            // 5. Deep Learning Foundations
            {
                step: 'Phase 10: 딥러닝 기초 (Neural Networks)',
                title: 'Deep Learning Foundations',
                description: '인공신경망의 수학적 원리와 역전파 알고리즘을 코드로 직접 구현해봅니다.',
                topics: ['Perceptron & Multi-Layer Perceptron (MLP)', 'Activation Functions (ReLU, Sigmoid, Tanh)', 'Backpropagation (Chain Rule application)', 'Loss Functions (MSE, Cross-Entropy)'],
                resources: [
                    { name: 'Neural Networks and Deep Learning', url: 'http://neuralnetworksanddeeplearning.com/' }
                ],
                quiz: {
                    question: '딥러닝에서 층(Layer)이 깊어질수록 기울기가 0에 가까워져 학습이 안 되는 현상은?',
                    options: ['Vanishing Gradient Problem', 'Exploding Gradient', 'Overfitting', 'Bias-Variance Tradeoff'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 최적화와 규제 (Optimization)',
                title: 'Training Deep Networks',
                description: '딥러닝 모델을 더 빠르고 안정적으로 학습시키기 위한 테크닉들을 다룹니다.',
                topics: ['Optimizers (Adam, RMSProp, SGD+Momentum)', 'Batch Normalization', 'Dropout & Regularization (L1/L2)', 'Learning Rate Scheduling'],
                resources: [
                    { name: 'cs231n - Optimization', url: 'https://cs231n.github.io/neural-networks-3/' }
                ],
                quiz: {
                    question: '학습 과정에서 각 배치의 입력을 정규화(평균 0, 분산 1)하여 학습 속도를 높이고 초기화 민감도를 줄이는 기법은?',
                    options: ['Batch Normalization', 'Dropout', 'Data Augmentation', 'Early Stopping'],
                    correctAnswer: 0
                }
            },
            // 6. Deep Learning Advanced (CV & NLP)
            {
                step: 'Phase 12: 컴퓨터 비전 (CNN)',
                title: 'Convolutional Neural Networks',
                description: '이미지의 공간적 특징을 추출하는 CNN 구조와 주요 아키텍처를 학습합니다.',
                topics: ['Convolution & Pooling Operations', 'ResNet & Skip Connections', 'Object Detection (YOLO, Faster R-CNN)', 'Image Segmentation (U-Net)'],
                resources: [
                    { name: 'Stanford CS231n', url: 'http://cs231n.stanford.edu/' }
                ],
                quiz: {
                    question: 'ResNet(Residual Network)에서 층이 깊어져도 학습이 잘 되도록 도입한 핵심 구조는?',
                    options: ['Skip Connection (Residual Block)', 'Inception Module', 'Dense Block', 'Attention Mechanism'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 13: 시퀀스 데이터 (RNN & LSTM)',
                title: 'Sequential Data Processing',
                description: '시간의 흐름이 있는 데이터(텍스트, 시계열)를 처리하는 모델을 배웁니다.',
                topics: ['RNN (Recurrent Neural Networks)', 'LSTM & GRU (Long-term dependencies)', 'Word Embeddings (Word2Vec, GloVe)', 'Seq2Seq Architecture'],
                resources: [
                    { name: 'Colah\'s Blog (LSTM)', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' }
                ],
                quiz: {
                    question: 'RNN의 장기 의존성(Long-term dependency) 문제를 해결하기 위해 Cell State와 Gate 구조를 도입한 모델은?',
                    options: ['LSTM', 'CNN', 'MLP', 'Autoencoder'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 14: 트랜스포머와 LLM (Attention)',
                title: 'Transformers & LLMs',
                description: '현대 AI의 표준이 된 Transformer 아키텍처와 대형언어모델(LLM)을 심도 있게 파헤칩니다.',
                topics: ['Self-Attention Mechanism', 'Transformer Encoder/Decoder (BERT vs GPT)', 'Positional Encoding', 'Instruction Tuning & RLHF'],
                resources: [
                    { name: 'The Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/' }
                ],
                quiz: {
                    question: 'Transformer 모델에서 입력 시퀀스의 모든 위치 간의 관계를 병렬적으로 계산하여 문맥을 파악하는 핵심 메커니즘은?',
                    options: ['Self-Attention', 'Convolution', 'Recurrence', 'MaxPooling'],
                    correctAnswer: 0
                }
            },
            // 7. MLOps & Production
            {
                step: 'Phase 15: 모델 패키징 (Docker)',
                title: 'Docker for ML',
                description: '단순 코드 실행이 아닌, 어디서든 실행 가능한 "컨테이너"로 모델을 포장합니다.',
                topics: ['Dockerfile 작성 최적화', 'Multi-stage Builds', 'Docker Compose', 'NVIDIA Container Toolkit (GPU Support)'],
                resources: [
                    { name: 'Docker Curriculum', url: 'https://docker-curriculum.com/' }
                ],
                quiz: {
                    question: 'Docker 이미지의 용량을 줄이고 보안을 강화하기 위해 빌드 과정과 실행 과정을 분리하는 기법은?',
                    options: ['Multi-stage Build', 'Docker Compose', 'Volume Mounting', 'Port Mapping'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 16: 모델 서빙 (Inference)',
                title: 'Model Serving & API',
                description: '학습된 모델을 웹 서비스나 애플리케이션에서 사용할 수 있도록 API 서버로 배포합니다.',
                topics: ['FastAPI (Async Inference)', 'TorchServe / TensorFlow Serving', 'gRPC vs REST', 'Dynamic Batching'],
                resources: [
                    { name: 'FastAPI Documentation', url: 'https://fastapi.tiangolo.com/' }
                ],
                quiz: {
                    question: '여러 개의 추론 요청을 모아서 한 번에 GPU로 처리하여 처리량(Throughput)을 높이는 기법은?',
                    options: ['Dynamic Batching', 'Model Quantization', 'Pruning', 'Caching'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 17: 오케스트레이션 (Kubernetes)',
                title: 'Kubernetes for ML',
                description: '대규모 트래픽을 처리하고 GPU 자원을 효율적으로 관리하기 위한 클러스터 운영을 배웁니다.',
                topics: ['Pods, Deployments, Services', 'KServe / Seldon Core', 'Horizontal Pod Autoscaling (HPA)', 'GPU Resource Scheduling'],
                resources: [
                    { name: 'Kubernetes Basics', url: 'https://kubernetes.io/docs/tutorials/kubernetes-basics/' }
                ],
                quiz: {
                    question: 'Kubernetes에서 애플리케이션의 배포 상태를 선언하고, 파드(Pod)의 복제본(Replicas) 개수를 유지 관리하는 리소스는?',
                    options: ['Deployment', 'Service', 'Ingress', 'ConfigMap'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 18: 파이프라인 자동화 (CI/CD/CT)',
                title: 'ML Pipelines & Automation',
                description: '데이터 전처리부터 학습, 배포까지 전 과정을 자동화하는 파이프라인을 구축합니다.',
                topics: ['Apache Airflow / Kubeflow Pipelines', 'GitHub Actions (CI/CD)', 'Continuous Training (CT)', 'Feature Store (Feast)'],
                resources: [
                    { name: 'MLOps: Continuous Delivery for ML', url: 'https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning' }
                ],
                quiz: {
                    question: '새로운 데이터가 들어오거나 성능이 떨어질 때 자동으로 모델을 재학습시키는 프로세스를 가리키는 용어는?',
                    options: ['Continuous Training (CT)', 'Continuous Integration (CI)', 'Continuous Deployment (CD)', 'Code Review'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 19: 모니터링과 유지보수',
                title: 'Monitoring & Observability',
                description: '배포된 모델이 잘 동작하는지 감시하고, 데이터가 변하는 현상(Drift)을 탐지합니다.',
                topics: ['Data Drift & Concept Drift', 'Prometheus & Grafana', 'Model Performance Monitoring', 'Outlier Detection'],
                resources: [
                    { name: 'Evidently AI (Drift Detection)', url: 'https://www.evidentlyai.com/' }
                ],
                quiz: {
                    question: '학습 데이터의 분포와 실제 서비스 입력 데이터의 분포가 달라져서 모델 성능이 저하되는 현상은?',
                    options: ['Data Drift (Covariate Shift)', 'Overfitting', 'Underfitting', 'Label Leakage'],
                    correctAnswer: 0
                }
            },
            // 8. Trends
            {
                step: 'Phase 20: 최신 트렌드 (GenAI & LLM Ops)',
                title: 'Generative AI & LLM Ops',
                description: '지금 가장 핫한 생성형 AI 튜닝과 운영 기술을 익혀 대체 불가능한 인재가 됩니다.',
                topics: ['RAG (Retrieval-Augmented Generation)', 'PEFT (LoRA, QLoRA)', 'Vector Database (Pinecone, Milvus)', 'LLM Evaluation (Ragas, LangSmith)'],
                resources: [
                    { name: 'LangChain Documentation', url: 'https://python.langchain.com/docs/get_started/introduction' }
                ],
                quiz: {
                    question: '거대 언어 모델(LLM)의 모든 파라미터를 튜닝하지 않고, 일부 파라미터만 학습시켜 효율적으로 미세조정(Fine-tuning)하는 기법은?',
                    options: ['PEFT (Parameter-Efficient Fine-Tuning)', 'Pre-training', 'Prompt Engineering', 'RAG'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 21: 커리어 & 포트폴리오',
                title: 'Career & Portfolio Strategy',
                description: '지금까지 배운 내용을 바탕으로 매력적인 포트폴리오를 만들고 면접을 준비합니다.',
                topics: ['End-to-End ML Project', 'Tech Blog Writing', 'Open Source Contribution', 'System Design Interview'],
                resources: [
                    { name: 'Machine Learning System Design Interview', url: 'https://github.com/chiphuyen/ml-system-design-case-studies' }
                ],
                quiz: {
                    question: 'ML 엔지니어 면접에서 "시스템 디자인" 질문이 나올 때 가장 중요하게 고려해야 할 사항이 아닌 것은?',
                    options: ['무조건 가장 최신의 복잡한 모델 사용하기', '요구사항 분석 및 메트릭 설정', '데이터 파이프라인 설계', '추론 지연시간(Latency)과 비용 고려'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '데이터 사이언티스트와 차이가 뭔가요?', answer: 'DS가 "원인 분석과 인사이트"에 집중한다면, ML 엔지니어는 "성능 좋은 예측 시스템 구축"에 더 집중합니다.' },
            { question: '어떤 수학이 필요한가요?', answer: '선형대수학(행렬 연산), 미적분(최적화), 확률통계(모델 평가)가 기본입니다.' }
        ]
    },
    {
        id: 'physical-ai',
        title: 'Physical AI Engineer',
        description: '인공지능의 지능을 실제 물리적인 몸체(로봇, 자동차, 드론 등)에 이식하여, 현실 세계에서 스스로 판단하고 움직이게 만드는 전문가입니다.',
        long_description: 'Physical AI Engineer는 단순 소프트웨어 개발을 넘어, AI(두뇌)와 하드웨어(몸)를 연결하여 물리 법칙이 지배하는 현실에서 임무를 수행하도록 설계하는 "현대판 로봇 조물주"입니다. 카메라, LiDAR 등으로 환경을 인지하고(Perception), 강화학습 등을 통해 최적의 행동을 결정하며(Decision), 이를 실제 로봇 팔이나 다리로 실행(Control)하는 전 과정을 다룹니다. 또한 위험한 현실 테스팅을 대신할 정교한 시뮬레이션(Digital Twin) 환경 구축 능력도 필수적입니다.',
        salary_range: '초봉 4,000 ~ 6,000만원',
        difficulty: 'Hard',
        demand: 'Medium',
        responsibilities: [
            '물리적 인지 및 환경 모델링 (Sensor Fusion & Perception)',
            '자율 행동 제어 알고리즘 설계 (Reinforcement Learning based Control)',
            'Sim-to-Real 가교 역할 (Simulation & Digital Twin)',
            '실시간 시스템 최적화 및 임베디드 AI 포팅'
        ],
        tech_stack: [
            { category: 'Language', skills: ['C++', 'Python'] },
            { category: 'Robotics', skills: ['ROS 2', 'Gazebo', 'Isaac Sim', 'Mujoco'] },
            { category: 'AI & Vision', skills: ['PyTorch', 'OpenCV', 'PCL', 'Reinforcement Learning'] },
            { category: 'Hardware', skills: ['NVIDIA Jetson', 'Raspberry Pi', 'Arduino', 'Sensors (LiDAR/IMU)'] }
        ],
        tags: ['Robotics', 'Embedded', 'Reinforcement Learning', 'Simulation'],
        focus_areas: [
            'Robotics Middleware (ROS 2)',
            'Deep Reinforcement Learning (DRL)',
            'Physics Simulation (Isaac Sim)',
            'Embedded AI Optimization'
        ],
        roadmap: [
            // 1. Foundations (Math & Physics)
            {
                step: 'Phase 1: 수학적 기초 (Math for Robotics)',
                title: 'Linear Algebra & Calculus',
                description: '로봇의 움직임을 기술하기 위한 필수 수학인 선형대수와 미적분을 익힙니다.',
                topics: ['Vectors & Matrices (Rotation Matrix)', 'Eigenvalues (PCA)', 'Calculus (Gradient Descent)', 'Optimization Basics'],
                resources: [
                    { name: 'Essence of Linear Algebra (3Blue1Brown)', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab' }
                ],
                quiz: {
                    question: '3차원 공간에서 물체의 회전을 표현할 때, 짐벌 락(Gimbal Lock) 현상을 피하기 위해 사용하는 수학적 도구는?',
                    options: ['Quaternion (쿼터니언)', 'Euler Angles (오일러 각)', 'Rotation Matrix', 'Vector Product'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 2: 물리학과 역학 (Physics & Dynamics)',
                title: 'Kinematics & Dynamics',
                description: '로봇 팔과 다리의 움직임을 계산하는 운동학(Kinematics)과 힘의 관계인 동역학(Dynamics)을 배웁니다.',
                topics: ['Forward/Inverse Kinematics (FK/IK)', 'Rigid Body Dynamics', 'Newton-Euler Equations', 'Jacobian Matrix'],
                resources: [
                    { name: 'Modern Robotics (Coursera)', url: 'https://www.coursera.org/specializations/modernrobotics' }
                ],
                quiz: {
                    question: '로봇 팔의 끝점(End-effector)의 위치를 알 때, 각 관절의 각도를 계산하는 과정을 무엇이라 하는가?',
                    options: ['Inverse Kinematics (역운동학)', 'Forward Kinematics (순운동학)', 'Dynamics', 'Motion Planning'],
                    correctAnswer: 0
                }
            },
            // 2. Programming Systems
            {
                step: 'Phase 3: 프로그래밍 기초 (C++ & Python)',
                title: 'C++ & Python for Robotics',
                description: '로봇 제어의 표준인 C++의 고성능 메모리 관리와 AI 모델링을 위한 Python을 마스터합니다.',
                topics: ['C++ OOP & STL', 'Python Interaction (pybind11)', 'Memory Management (Pointers)', 'Real-time Constraints'],
                resources: [
                    { name: 'Learn C++', url: 'https://www.learncpp.com/' }
                ],
                quiz: {
                    question: 'C++에서 메모리 누수를 방지하기 위해 사용이 권장되는, 자동으로 메모리를 해제해주는 포인터는?',
                    options: ['Smart Pointer (shared_ptr, unique_ptr)', 'Raw Pointer', 'Null Pointer', 'Void Pointer'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 4: 리눅스 시스템 (Linux & OS)',
                title: 'Linux Environment',
                description: '로봇 개발의 주 무대인 리눅스 환경과 쉘 스크립팅, 파일 시스템을 익힙니다.',
                topics: ['Ubuntu/Debian Basics', 'Shell Scripting (Bash)', 'Process Management', 'Networking Basics (SSH, IP)'],
                resources: [
                    { name: 'Linux Command Line Basics', url: 'https://ubuntu.com/tutorials/command-line-for-beginners' }
                ],
                quiz: {
                    question: '리눅스 터미널에서 현재 실행 중인 프로세스들의 상태를 실시간으로 확인하는 명령어는?',
                    options: ['top (or htop)', 'ls', 'cd', 'grep'],
                    correctAnswer: 0
                }
            },
            // 3. Robotics Core
            {
                step: 'Phase 5: 로봇 미들웨어 (ROS 2 Basic)',
                title: 'ROS 2 Fundamentals',
                description: '로봇 소프트웨어 개발의 표준 운영체제인 ROS 2의 기본 개념과 통신 방식을 배웁니다.',
                topics: ['Nodes & Lifecycle', 'Topics (Pub/Sub)', 'Services (Req/Res)', 'Actions (Long-running tasks)'],
                resources: [
                    { name: 'ROS 2 Humble Documentation', url: 'https://docs.ros.org/en/humble/' }
                ],
                quiz: {
                    question: 'ROS 2에서 노드 간에 데이터를 지속적으로 스트리밍(방송)할 때 사용하는 통신 방식은?',
                    options: ['Topic (토픽)', 'Service (서비스)', 'Action (액션)', 'Parameter'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 6: ROS 2 심화 (Advanced ROS 2)',
                title: 'Advanced ROS 2 Features',
                description: '복잡한 로봇 시스템을 효율적으로 관리하기 위한 ROS 2의 고급 기능을 다룹니다.',
                topics: ['Custom Interfaces (.msg/.srv)', 'DDS (Data Distribution Service)', 'Launch Files', 'ROS 2 Parameters'],
                resources: [
                    { name: 'The Construct ROS Courses', url: 'https://www.theconstructsim.com/' }
                ],
                quiz: {
                    question: 'ROS 2가 실시간성과 신뢰성을 확보하기 위해 채택한 통신 미들웨어 표준은?',
                    options: ['DDS (Data Distribution Service)', 'TCP/IP', 'HTTP', 'WebSocket'],
                    correctAnswer: 0
                }
            },
            // 4. Perception
            {
                step: 'Phase 7: 컴퓨터 비전 기초 (Computer Vision)',
                title: 'Computer Vision Basics',
                description: '로봇의 눈이 되는 카메라 데이터를 처리하고 분석하는 기술을 배웁니다.',
                topics: ['Image Processing (OpenCV)', 'Feature Extraction (SIFT, ORB)', 'Camera Calibration', 'Hough Transform'],
                resources: [
                    { name: 'OpenCV Tutorials', url: 'https://docs.opencv.org/4.x/d9/df8/tutorial_root.html' }
                ],
                quiz: {
                    question: '카메라 렌즈의 왜곡을 보정하고, 3차원 공간의 점을 2차원 이미지로 매핑하기 위해 구해야 하는 파라미터 행렬은?',
                    options: ['Intrinsic Matrix (내부 파라미터)', 'Rotation Matrix', 'Translation Vector', 'Identity Matrix'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 8: 3D 비전과 포인트 클라우드',
                title: '3D Perception & LiDAR',
                description: 'LiDAR나 RGB-D 카메라로 얻은 3차원 점군(Point Cloud) 데이터를 처리합니다.',
                topics: ['Point Cloud Library (PCL)', 'Depth Estimation', '3D Object Detection', 'PointNet Architecture'],
                resources: [
                    { name: 'PCL Documentation', url: 'https://pointclouds.org/documentation/' }
                ],
                quiz: {
                    question: '3D 공간상의 점들의 집합(Point Cloud)을 처리할 때, 노이즈를 제거하거나 다운샘플링하기 위해 사용하는 대표적인 필터는?',
                    options: ['Voxel Grid Filter', 'Low Pass Filter', 'Sobel Filter', 'Canny Edge Detector'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 9: 위치 추정과 지도 작성 (SLAM)',
                title: 'SLAM (Simultaneous Localization and Mapping)',
                description: '로봇이 낯선 환경에서 자신의 위치를 파악하고 지도를 그리는 핵심 기술입니다.',
                topics: ['Kalman Filter / Particle Filter', 'Visual SLAM (ORB-SLAM)', 'LiDAR SLAM (Cartographer)', 'Loop Closure Detection'],
                resources: [
                    { name: 'SLAM for Dummies', url: 'https://ocw.mit.edu/courses/16-412j-cognitive-robotics-spring-2016/resources/session-14-slam/' }
                ],
                quiz: {
                    question: '로봇이 이전에 방문했던 장소를 다시 방문했음을 인식하여, 누적된 위치 오차를 획기적으로 줄이는 기술은?',
                    options: ['Loop Closure (루프 결합)', 'Dead Reckoning', 'Odometry', 'Path Planning'],
                    correctAnswer: 0
                }
            },
            // 5. Control & Planning
            {
                step: 'Phase 10: 제어 이론 (Control Theory)',
                title: 'Classic Control',
                description: '로봇이 목표 상태로 정확하고 안정적으로 도달하도록 하는 제어 기법을 익힙니다.',
                topics: ['PID Control', 'LQR (Linear Quadratic Regulator)', 'State Space Model', 'Stability Analysis'],
                resources: [
                    { name: 'Control Bootcamp (Steve Brunton)', url: 'https://www.youtube.com/playlist?list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m' }
                ],
                quiz: {
                    question: 'PID 제어기에서 "현재 오차"에 비례하여 제어 입력을 조절하는 항은?',
                    options: ['P (Proportional)', 'I (Integral)', 'D (Derivative)', 'Gain'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 11: 경로 계획 (Path Planning)',
                title: 'Path & Motion Planning',
                description: '장애물을 피해서 목적지까지 가는 최적의 경로를 생성하는 알고리즘을 배웁니다.',
                topics: ['A* Algorithm', 'RRT / RRT* (Rapidly-exploring Random Tree)', 'Navigation Stack (Nav2)', 'Trajectory Optimization'],
                resources: [
                    { name: 'Introduction to A*', url: 'https://www.redblobgames.com/pathfinding/a-star/introduction.html' }
                ],
                quiz: {
                    question: '고차원 공간이나 복잡한 환경에서 무작위 샘플링을 통해 빠르게 경로를 찾아내는 확률적 경로 계획 알고리즘은?',
                    options: ['RRT (Rapidly-exploring Random Tree)', 'A*', 'Dijkstra', 'BFS'],
                    correctAnswer: 0
                }
            },
            // 6. AI & Reinforcement Learning
            {
                step: 'Phase 12: 딥러닝 기초 (Deep Learning)',
                title: 'Deep Learning for Robotics',
                description: '로봇의 인지 및 제어에 사용되는 기본적인 딥러닝 모델을 학습합니다.',
                topics: ['Neural Networks Basics', 'CNN (for Vision)', 'PyTorch Framework', 'Inference Optimization'],
                resources: [
                    { name: 'Deep Learning Specialization', url: 'https://www.coursera.org/specializations/deep-learning' }
                ],
                quiz: {
                    question: '이미지 데이터를 처리하여 객체를 인식하는 데 가장 효과적인 신경망 구조는?',
                    options: ['CNN (Convolutional Neural Network)', 'RNN', 'MLP', 'Transformer'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 13: 강화학습 기초 (RL Basics)',
                title: 'Reinforcement Learning Foundations',
                description: '보상을 통해 시행착오를 겪으며 스스로 학습하는 강화학습의 원리를 배웁니다.',
                topics: ['Markov Decision Process (MDP)', 'Q-Learning', 'Policy Gradients', 'Exploration vs Exploitation'],
                resources: [
                    { name: 'Spinning Up in Deep RL (OpenAI)', url: 'https://spinningup.openai.com/' }
                ],
                quiz: {
                    question: '강화학습에서 에이전트가 어떤 상태(State)에서 취할 행동(Action)을 결정하는 규칙이나 전략을 무엇이라 하는가?',
                    options: ['Policy (정책)', 'Reward (보상)', 'Environment (환경)', 'Episode (에피소드)'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 14: 심화 강화학습 (Deep RL)',
                title: 'Deep Reinforcement Learning',
                description: '딥러닝과 강화학습을 결합하여 복잡한 로봇 제어 문제를 해결합니다.',
                topics: ['DQN (Deep Q-Network)', 'PPO (Proximal Policy Optimization)', 'SAC (Soft Actor-Critic)', 'Reward Shaping'],
                resources: [
                    { name: 'Hugging Face Deep RL Course', url: 'https://huggingface.co/learn/deep-rl-course/unit1/introduction' }
                ],
                quiz: {
                    question: '연속적인 행동 공간(Continuous Action Space)을 가지는 로봇 제어 문제에 적합한 최신 강화학습 알고리즘은?',
                    options: ['PPO / SAC', 'DQN', 'Tabular Q-Learning', 'Genetic Algorithm'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 15: 모방 학습 (Imitation Learning)',
                title: 'Imitation Learning',
                description: '전문가의 시범 데이터를 통해 로봇이 빠르게 동작을 배우는 기법입니다.',
                topics: ['Behavior Cloning', 'Inverse Reinforcement Learning', 'DAgger', 'Demonstration Data Collection'],
                resources: [
                    { name: 'Imitation Learning Tutorial', url: 'https://sites.google.com/view/icml2018-imitation-learning/' }
                ],
                quiz: {
                    question: '전문가(사람)가 시연한 행동을 지도 학습(Supervised Learning) 방식으로 그대로 따라 하도록 학습하는 기법은?',
                    options: ['Behavior Cloning', 'Q-Learning', 'Policy Gradient', 'Monte Carlo'],
                    correctAnswer: 0
                }
            },
            // 7. Simulation & Sim2Real
            {
                step: 'Phase 16: 물리 시뮬레이션 (Physics Engines)',
                title: 'Robotics Simulation',
                description: '현실과 유사한 가상 환경을 구축하여 AI를 안전하게 학습시킵니다.',
                topics: ['Gazebo (Classic/Ignition)', 'NVIDIA Isaac Sim', 'MuJoCo', 'URDF/SDF Modeling'],
                resources: [
                    { name: 'NVIDIA Isaac Sim', url: 'https://developer.nvidia.com/isaac-sim' }
                ],
                quiz: {
                    question: 'NVIDIA Omniverse 기반으로 만들어진, 고품질 렌더링과 물리 엔진을 지원하는 최신 로봇 시뮬레이터는?',
                    options: ['Isaac Sim', 'Gazebo 9', 'V-Rep', 'Webots'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 17: Sim-to-Real Transfer',
                title: 'Sim-to-Real Transfer',
                description: '시뮬레이션에서 학습한 모델을 현실 세계에서도 잘 작동하도록 만드는 기술입니다.',
                topics: ['Domain Randomization', 'System Identification', 'Reality Gap Analysis', 'Robust Control'],
                resources: [
                    { name: 'OpenAI Solving Rubiks Cube', url: 'https://openai.com/research/solving-rubiks-cube' }
                ],
                quiz: {
                    question: '시뮬레이션 환경의 물리 파라미터(마찰력, 질량 등)나 시각적 요소(조명, 텍스처)를 무작위로 변화시켜, 모델의 적응력을 높이는 기법은?',
                    options: ['Domain Randomization', 'Data Augmentation', 'Dropout', 'Fine-tuning'],
                    correctAnswer: 0
                }
            },
            // 8. Deployment & Hardware
            {
                step: 'Phase 18: 임베디드 AI (Edge AI)',
                title: 'Edge AI Deployment',
                description: '학습된 무거운 AI 모델을 소형 컴퓨터(Edge Device)에서 실시간으로 돌아가게 경량화합니다.',
                topics: ['Model Quantization', 'TensorRT Optimization', 'TFLite / ONNX Runtime', 'Pruning'],
                resources: [
                    { name: 'NVIDIA TensorRT', url: 'https://developer.nvidia.com/tensorrt' }
                ],
                quiz: {
                    question: 'AI 모델의 파라미터 정밀도를 32비트(FP32)에서 8비트(INT8) 등으로 낮추어, 정확도 손실을 최소화하면서 속도를 높이는 기술은?',
                    options: ['Quantization (양자화)', 'Distillation', 'Compilation', 'Overclocking'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 19: 하드웨어 인터페이스',
                title: 'Hardware Interfacing',
                description: '센서와 액추에이터를 코드로 직접 제어하는 하드웨어 통신을 다룹니다.',
                topics: ['GPIO / PWM Control', 'Serial (UART), I2C, SPI', 'Motor Drivers', 'Microcontroller (Arduino/STM32)'],
                resources: [
                    { name: 'Arduino Reference', url: 'https://www.arduino.cc/reference/en/' }
                ],
                quiz: {
                    question: '마이크로컨트롤러와 센서 간에 데이터를 주고받을 때 사용하는 2선식(SDA, SCL) 직렬 통신 프로토콜은?',
                    options: ['I2C', 'SPI', 'UART', 'Ethernet'],
                    correctAnswer: 0
                }
            },
            // 9. Career
            {
                step: 'Phase 20: 프로젝트 & 포트폴리오',
                title: 'Capstone Project',
                description: '실제 로봇(또는 고정밀 시뮬레이션)을 사용하여 나만의 Physical AI 프로젝트를 완성합니다.',
                topics: ['System Integration', 'Real-world Testing', 'Demo Video Production', 'Open Source Contribution'],
                resources: [
                    { name: 'ROS 2 Projects', url: 'https://roboticsbackend.com/' }
                ],
                quiz: {
                    question: '로봇 프로젝트를 진행할 때, 전체 시스템의 데이터 흐름과 노드 간의 관계를 시각화하여 디버깅을 돕는 ROS 도구는?',
                    options: ['rqt_graph', 'rviz', 'gazebo', 'ros2 topic echo'],
                    correctAnswer: 0
                }
            },
            {
                step: 'Phase 21: 커리어 & 트렌드',
                title: 'Future of Physical AI',
                description: '휴머노이드, 자율주행 등 최신 트렌드를 파악하고 커리어 방향을 설정합니다.',
                topics: ['Humanoid Robots', 'End-to-End Autonomous Driving', 'Foundation Models for Robotics (VLA)', 'Research Papers (ICRA/IROS)'],
                resources: [
                    { name: 'IEEE Spectrum Robotics', url: 'https://spectrum.ieee.org/topic/robotics/' }
                ],
                quiz: {
                    question: '최근 로봇 분야에서 주목받는, "비전(Vision)", "언어(Language)", "행동(Action)"을 통합하여 학습한 거대 모델을 무엇이라 하는가?',
                    options: ['VLA (Vision-Language-Action) Model', 'LLM (Large Language Model)', 'GAN', 'Expert System'],
                    correctAnswer: 0
                }
            }
        ],
        faq: [
            { question: '하드웨어를 직접 만들어야 하나요?', answer: '아니요, 주로 기구 설계는 기계공학 엔지니어가 하고, 여러분은 그 "머리(지능)"를 만드는 소프트웨어에 집중합니다. 다만 회로도는 볼 줄 알면 좋습니다.' },
            { question: 'C++이 필수인가요?', answer: '실시간 제어와 성능이 중요한 로봇 분야에서는 C++이 필수입니다. AI 모델링은 Python으로 하더라도 배포는 C++로 하는 경우가 많습니다.' }
        ]
    },
];

export const QUESTIONS: Question[] = [
    // 1. 기본 성향 (개발 vs 비개발)
    {
        id: 1,
        question: '다음 중 가장 흥미를 느끼는 작업은 무엇인가요?',
        options: [
            { text: '직접 코드를 짜서 무언가 만들어내는 것', weights: { 'ai-app': 3, 'data-eng': 3, 'mlops': 3, 'physical-ai': 3, 'research': 2 } },
            { text: '데이터를 보고 숨겨진 의미를 찾아내는 것', weights: { 'data-sci': 5, 'prompt-eng': 3 } },
            { text: '사람들의 문제를 정의하고 해결책을 기획하는 것', weights: { 'pm': 5, 'ml-eng': 4, 'prompt-eng': 2 } },
        ],
    },
    // 2. 개발 선호도 (서비스 vs 인프라)
    {
        id: 2,
        question: '코딩을 한다면 어떤 스타일을 선호하나요?',
        options: [
            { text: '눈에 보이는 결과물(웹/앱)을 빠르게 만드는 것', weights: { 'ai-app': 5, 'prompt-eng': 2 } },
            { text: '보이지 않는 곳에서 대용량 처리를 안정적으로 하는 것', weights: { 'data-eng': 5, 'mlops': 5, 'ml-eng': 2 } },
            { text: '복잡한 수학적 알고리즘을 구현하는 것', weights: { 'research': 5, 'data-sci': 4 } },
            { text: '코딩보다는 논리적인 글쓰기나 구조 설계가 좋다', weights: { 'pm': 4, 'prompt-eng': 4 } },
        ],
    },
    // 3. AI 모델에 대한 태도
    {
        id: 3,
        question: 'AI 모델을 다룰 때 어떤 점이 즐거운가요?',
        options: [
            { text: '최신 모델(GPT-4 등)을 API로 가져와서 뚝딱 서비스를 만드는 것', weights: { 'ai-app': 5, 'pm': 2 } },
            { text: '어떤 명령어를 입력해야 AI가 찰떡같이 알아듣는지 실험하는 것', weights: { 'prompt-eng': 5, 'pm': 2 } },
            { text: '모델의 내부 구조(Transformer 등)를 뜯어보고 이해하는 것', weights: { 'research': 5, 'data-sci': 3 } },
            { text: '모델이 하루 100만 번 호출되어도 죽지 않게 만드는 것', weights: { 'mlops': 5, 'ml-eng': 3 } },
        ],
    },
    // 4. 데이터 핸들링
    {
        id: 4,
        question: '더러운 데이터(중복, 결측치 등)를 만났을 때 반응은?',
        options: [
            { text: '집요하게 파고들어 깨끗하게 정제하고 싶다.', weights: { 'data-eng': 4, 'data-sci': 5 } },
            { text: '데이터 정제는 너무 귀찮다. 잘 정제된 데이터만 쓰고 싶다.', weights: { 'ai-app': 3, 'prompt-eng': 3, 'research': 2 } },
            { text: '이 데이터로 어떤 가치를 만들 수 있을지부터 고민한다.', weights: { 'pm': 4, 'ml-eng': 4 } },
        ],
    },
    // 5. 인프라/클라우드
    {
        id: 5,
        question: '리눅스(Linux) 터미널과 클라우드(AWS/Docker) 화면을 보면 어떤가요?',
        options: [
            { text: '검은 화면에 흰 글씨.. 뭔가 있어 보이고 재밌다.', weights: { 'mlops': 5, 'data-eng': 4, 'physical-ai': 4 } },
            { text: '필요하면 쓰지만, 가능하면 피하고 싶다.', weights: { 'data-sci': 3, 'ai-app': 2, 'research': 2 } },
            { text: '전혀 모르겠다. 그냥 버튼 누르면 되면 좋겠다.', weights: { 'pm': 4, 'prompt-eng': 3 } },
        ],
    },
    // 6. 소통/비즈니스
    {
        id: 6,
        question: '팀 프로젝트에서 갈등이 생기면 주로 어떤 역할을 하나요?',
        options: [
            { text: '내 의견을 논리적으로 설득해서 관철시킨다.', weights: { 'ml-eng': 3, 'research': 3 } },
            { text: '상대방의 의견을 듣고 절충안(중재)을 제안한다.', weights: { 'pm': 5, 'ml-eng': 4 } },
            { text: '말보다는 묵묵히 내 할 일(코딩)을 해서 기여한다.', weights: { 'ai-app': 3, 'data-eng': 3, 'mlops': 3 } },
        ],
    },
    // 7. 문제 해결 접근 방식
    {
        id: 7,
        question: '막히는 문제가 생겼을 때 해결 방식은?',
        options: [
            { text: '최신 논문이나 해외 기술 블로그를 깊이 파본다.', weights: { 'research': 5, 'data-sci': 4, 'physical-ai': 3 } },
            { text: 'Stack Overflow나 공식 문서를 찾아서 빠르게 적용한다.', weights: { 'ai-app': 5, 'data-eng': 3, 'physical-ai': 3 } },
            { text: '여러 가지 입력값을 바꿔가며 될 때까지 실험해본다.', weights: { 'prompt-eng': 5, 'mlops': 2 } },
            { text: '동료나 전문가에게 물어보고 구조적인 해결책을 찾는다.', weights: { 'ml-eng': 5, 'pm': 3 } },
        ],
    },
    // 8. 수학/통계
    {
        id: 8,
        question: '학창 시절 확률과 통계, 미적분 수업 시간은 어땠나요?',
        options: [
            { text: '수식이 주는 명쾌함이 좋았다.', weights: { 'research': 5, 'data-sci': 5, 'physical-ai': 4 } },
            { text: '필요성은 알지만 좋아하진 않았다.', weights: { 'mlops': 2, 'data-eng': 2, 'ai-app': 2 } },
            { text: '숫자만 봐도 머리가 아팠다.', weights: { 'pm': 2, 'prompt-eng': 3, 'ai-app': 1 } },
        ],
    },
    // 9. 결과물의 형태
    {
        id: 9,
        question: '내가 만든 결과물, 어떤 형태일 때 가장 뿌듯한가요?',
        options: [
            { text: '사용자가 "우와 신기해요!"라고 반응하는 웹 서비스', weights: { 'ai-app': 5, 'prompt-eng': 3 } },
            { text: '오차율이 0.1% 줄어든 고성능 모델', weights: { 'research': 5, 'data-sci': 4 } },
            { text: '장애 없이 24시간 매끄럽게 돌아가는 시스템', weights: { 'mlops': 5, 'data-eng': 4 } },
            { text: '알아서 척척 정리된 깔끔한 기획서/보고서', weights: { 'pm': 5, 'ml-eng': 4 } },
        ],
    },
    // 10. 관심 기술 스택
    {
        id: 10,
        question: '다음 중 가장 배워보고 싶은 기술은?',
        options: [
            { text: 'Kubernetes, Docker, CI/CD', weights: { 'mlops': 5, 'data-eng': 4, 'physical-ai': 3 } },
            { text: 'React, Next.js, FastAPI', weights: { 'ai-app': 5 } },
            { text: 'PyTorch, TensorFlow, Hugging Face', weights: { 'research': 4, 'data-sci': 4, 'physical-ai': 3 } },
            { text: 'Tableau, SQL, Pandas', weights: { 'data-sci': 5, 'data-eng': 3 } },
        ],
    },
    // 11. 반복 작업
    {
        id: 11,
        question: '반복적인 작업(노가다)을 해야 할 때?',
        options: [
            { text: '스크립트를 짜서 자동으로 처리되게 만든다.', weights: { 'data-eng': 5, 'mlops': 5, 'ai-app': 3 } },
            { text: 'AI에게 시켜서 효율적으로 처리한다.', weights: { 'prompt-eng': 5, 'pm': 3 } },
            { text: '그냥 묵묵히 한다. 꼼꼼함이 중요하다.', weights: { 'data-sci': 2 } },
        ],
    },
    // 12. 새로운 툴
    {
        id: 12,
        question: '새로운 AI 툴(ChatGPT, Midjourney 등)이 나오면?',
        options: [
            { text: '이걸로 어떤 사업을 할 수 있을지 구상한다.', weights: { 'pm': 5, 'ml-eng': 4 } },
            { text: '어떻게 프롬프트를 쳐야 잘 나오는지 연구한다.', weights: { 'prompt-eng': 5 } },
            { text: 'API 문서부터 찾아서 연동해본다.', weights: { 'ai-app': 5 } },
            { text: '어떤 원리로 돌아가는지 기술 블로그를 찾아본다.', weights: { 'research': 3, 'ml-eng': 3 } },
        ],
    },
    // 13. 발표 부담
    {
        id: 13,
        question: '발표(Presentation)에 대한 부담감은?',
        options: [
            { text: '남들 앞에서 설명하고 설득하는 게 즐겁다.', weights: { 'pm': 5, 'ml-eng': 5, 'data-sci': 3 } },
            { text: '준비하면 할 수 있지만 떨린다.', weights: { 'ai-app': 2, 'prompt-eng': 2 } },
            { text: '발표보다는 글로 정리해서 보여주고 싶다.', weights: { 'research': 2, 'data-eng': 3, 'mlops': 3 } },
        ],
    },
    // 14. SQL 쿼리
    {
        id: 14,
        question: '하루 종일 SQL 쿼리만 짜야 한다면?',
        options: [
            { text: '데이터 뽑는 재미가 있어서 괜찮다.', weights: { 'data-eng': 5, 'data-sci': 4 } },
            { text: '너무 지루할 것 같다.', weights: { 'ai-app': 3, 'mlops': 3, 'research': 3 } },
            { text: '필요한 데이터라면 기꺼이 한다.', weights: { 'pm': 2, 'ml-eng': 2 } },
        ],
    },
    // 15. 고객 미팅
    {
        id: 15,
        question: '고객사 미팅에 나가야 한다면?',
        options: [
            { text: '기술적인 부분을 잘 설명해줄 자신이 있다.', weights: { 'ml-eng': 5, 'pm': 3 } },
            { text: '비즈니스 요구사항을 듣고 정리하는 게 좋다.', weights: { 'pm': 5 } },
            { text: '가능하면 개발팀 내부 회의만 하고 싶다.', weights: { 'ai-app': 2, 'mlops': 3, 'data-eng': 3 } },
        ],
    },
    // 16. 중요 가치
    {
        id: 16,
        question: '가장 중요하게 생각하는 가치는?',
        options: [
            { text: '혁신(Innovation)과 새로운 발견', weights: { 'research': 5, 'ai-app': 3 } },
            { text: '안정성(Stability)과 신뢰', weights: { 'mlops': 5, 'data-eng': 5, 'ml-eng': 3 } },
            { text: '실용성(Utility)과 사용자 가치', weights: { 'pm': 5, 'ai-app': 4, 'prompt-eng': 3 } },
        ],
    },
    // 17. 코드 에러
    {
        id: 17,
        question: '코드가 에러가 났을 때?',
        options: [
            { text: '로그를 한 줄 한 줄 뜯어보며 원인을 찾는다.', weights: { 'mlops': 4, 'data-eng': 4, 'backend': 3 } },
            { text: '에러 메시지를 복사해서 AI에게 물어본다.', weights: { 'ai-app': 3, 'prompt-eng': 3 } },
            { text: '구조적으로 어디가 잘못됐는지 전체 그림을 본다.', weights: { 'ml-eng': 4, 'research': 3 } },
        ],
    },
    // 18. 협업 툴
    {
        id: 18,
        question: '협업 툴(Jira, Slack) 사용 숙련도는?',
        options: [
            { text: '알림 설정을 칼같이 하고 티켓 관리를 잘한다.', weights: { 'pm': 5, 'mlops': 3 } },
            { text: '필요한 기능만 쓴다.', weights: { 'ai-app': 2, 'data-sci': 2 } },
            { text: '이런 툴보다는 직접 대화하는 게 편하다.', weights: { 'ml-eng': 2 } },
        ],
    },
    // 19. MBTI (재미)
    {
        id: 19,
        question: '본인의 MBTI와 가까운 것은? (재미용)',
        options: [
            { text: '계획적이고 체계적인 J형', weights: { 'pm': 3, 'mlops': 3, 'data-eng': 3 } },
            { text: '유연하고 즉흥적인 P형', weights: { 'research': 3, 'ai-app': 3, 'prompt-eng': 3 } },
            { text: '논리적인 T형', weights: { 'data-sci': 3, 'ml-eng': 3 } },
            { text: '공감하는 F형', weights: { 'pm': 3, 'prompt-eng': 3 } },
        ],
    },
    // 20. 3년 뒤 모습
    {
        id: 20,
        question: '마지막으로, 당신이 꿈꾸는 3년 뒤 모습은?',
        options: [
            { text: '나만의 AI 서비스를 런칭한 창업가', weights: { 'ai-app': 4, 'pm': 4 } },
            { text: '대규모 시스템을 지탱하는 기술 리더', weights: { 'ml-eng': 5, 'mlops': 4, 'data-eng': 4 } },
            { text: '세계적인 학회에 이름을 올린 AI 연구자', weights: { 'research': 5 } },
            { text: '복잡한 비즈니스 문제를 데이터로 풀어내는 전문가', weights: { 'data-sci': 5, 'prompt-eng': 3 } },
        ],
    },
    // 21. 하드웨어/로봇 (New for Physical AI)
    {
        id: 21,
        question: '개발하고 싶은 대상이 무엇인가요?',
        options: [
            { text: '모니터 속의 소프트웨어 (웹, 앱, 서버)', weights: { 'ai-app': 4, 'pm': 2, 'data-sci': 2 } },
            { text: '현실 세계에서 움직이는 기계 (로봇, 드론, 자동차)', weights: { 'physical-ai': 5, 'research': 2 } },
            { text: '데이터 그 자체 (숫자와 통계)', weights: { 'data-sci': 4, 'data-eng': 3 } },
            { text: '거대하고 복잡한 시스템 인프라', weights: { 'mlops': 4, 'data-eng': 4 } },
        ],
    },
];

export const QUESTIONS_BEGINNER: Question[] = [
    // 1. 흥미 (만들기 vs 분석하기)
    {
        id: 1,
        question: '무언가를 배울 때 더 재미있는 것은?',
        options: [
            { text: '화면에 그림이나 글자가 짠! 하고 나타나는 것 (웹사이트 만들기)', weights: { 'ai-app': 4, 'prompt-eng': 3, 'physical-ai': 3 } },
            { text: '복잡한 데이터를 정리해서 깔끔한 표로 만드는 것', weights: { 'data-sci': 4, 'data-eng': 3 } },
            { text: '사람들이 왜 이걸 불편해하는지 이유를 찾는 것', weights: { 'pm': 5, 'ml-eng': 2 } },
        ],
    },
    // 2. 컴퓨터 다루기
    {
        id: 2,
        question: '검은색 화면(터미널)에 명령어를 치는 것이...',
        options: [
            { text: '뭔가 해커 같고 멋있다. 더 배우고 싶다.', weights: { 'mlops': 5, 'data-eng': 4, 'ml-eng': 3, 'physical-ai': 4 } },
            { text: '아직은 낯설고 무섭다. 버튼이 편하다.', weights: { 'pm': 4, 'ai-app': 3, 'prompt-eng': 3 } },
            { text: '필요하다면 배울 수 있다.', weights: { 'research': 3, 'data-sci': 3 } },
        ],
    },
    // 3. 문제 해결
    {
        id: 3,
        question: '친구가 컴퓨터가 고장 났다고 물어보면?',
        options: [
            { text: '어디가 문제인지 하나씩 뜯어서 고쳐준다.', weights: { 'mlops': 4, 'ml-eng': 3, 'physical-ai': 5 } },
            { text: '인터넷에 검색해서 해결 방법을 찾아 보내준다.', weights: { 'ai-app': 4, 'data-eng': 3 } },
            { text: '새로 사는 게 낫지 않아? 라고 조언한다.', weights: { 'pm': 3 } },
        ],
    },
    // 4. 수업 시간
    {
        id: 4,
        question: '수학 시간이나 통계학 수업 때...',
        options: [
            { text: '알쏭달쏭한 수수께끼를 푸는 것 같아 재밌었다.', weights: { 'research': 5, 'data-sci': 5, 'physical-ai': 4 } },
            { text: '공식 외우는 게 너무 싫었다.', weights: { 'ai-app': 3, 'pm': 3, 'prompt-eng': 3 } },
            { text: '답이 딱 떨어지는 게 좋았다.', weights: { 'data-eng': 4, 'mlops': 3 } },
        ],
    },
    // 5. 챗봇 사용
    {
        id: 5,
        question: 'ChatGPT 같은 AI랑 대화할 때...',
        options: [
            { text: '어떻게 말해야 얘가 더 똑똑하게 대답할지 고민한다.', weights: { 'prompt-eng': 5, 'ai-app': 3 } },
            { text: '신기하긴 한데, 이게 어떻게 작동하는지 원리가 궁금하다.', weights: { 'research': 4, 'ml-eng': 4 } },
            { text: '이걸로 숙제를 하거나 일을 편하게 할 방법을 찾는다.', weights: { 'pm': 4, 'ai-app': 3 } },
        ],
    },
    // 6. 꼼꼼함 vs 속도
    {
        id: 6,
        question: '과제를 제출해야 할 때 나의 스타일은?',
        options: [
            { text: '조금 늦더라도 완벽하게 검토해서 낸다.', weights: { 'data-sci': 4, 'research': 4, 'mlops': 3 } },
            { text: '일단 완성해서 빨리 제출하고 쉰다.', weights: { 'ai-app': 4, 'pm': 3, 'data-eng': 2 } },
        ],
    },
    // 7. 관심 분야
    {
        id: 7,
        question: 'IT 뉴스에서 가장 눈길이 가는 제목은?',
        options: [
            { text: '"AI가 그린 그림이 미술 대회 우승"', weights: { 'ai-app': 3, 'prompt-eng': 4 } },
            { text: '"카카오톡 서버가 멈춘 이유와 해결 과정"', weights: { 'mlops': 5, 'data-eng': 4 } },
            { text: '"새로운 AI 기술 논문 발표"', weights: { 'research': 5, 'ml-eng': 4 } },
            { text: '"올해 가장 많이 팔린 IT 서비스 트렌드"', weights: { 'pm': 5, 'data-sci': 3 } },
        ],
    },
    // 8. 팀플 역할
    {
        id: 8,
        question: '조별 과제를 할 때 내가 주로 하는 말은?',
        options: [
            { text: '"자, 우리 이렇게 역할을 나누고 언제까지 하자."', weights: { 'pm': 5, 'mlops': 2 } },
            { text: '"내가 자료 조사랑 정리를 싹 다 할게."', weights: { 'data-eng': 4, 'data-sci': 3 } },
            { text: '"PPT 디자인이랑 발표는 내가 할게."', weights: { 'ai-app': 3, 'prompt-eng': 3 } },
            { text: '"어려운 부분 있으면 내가 좀 도와줄게."', weights: { 'ml-eng': 4, 'research': 3 } },
        ],
    },
    // 9. 정리 vs 창조
    {
        id: 9,
        question: '책상은 어떤 상태인가요?',
        options: [
            { text: '모든 물건이 각 잡혀서 정해진 위치에 있다.', weights: { 'data-eng': 4, 'mlops': 4, 'pm': 3 } },
            { text: '정신없어 보이지만 나만의 규칙이 있다.', weights: { 'ai-app': 4, 'research': 4 } },
            { text: '치워야지 생각만 하고 계속 쌓인다.', weights: { 'prompt-eng': 3, 'data-sci': 2 } },
        ],
    },
    // 10. 새로운 기술 습득
    {
        id: 10,
        question: '새로운 핸드폰이나 기계를 샀을 때?',
        options: [
            { text: '설명서부터 꼼꼼히 읽어본다.', weights: { 'mlops': 4, 'data-eng': 3 } },
            { text: '일단 이것저것 눌러보면서 기능을 익힌다.', weights: { 'ai-app': 5, 'prompt-eng': 3 } },
            { text: '유튜브 리뷰 영상을 찾아본다.', weights: { 'pm': 3, 'data-sci': 2 } },
        ],
    },
    // 11. 친구의 고민 상담
    {
        id: 11,
        question: '친구가 힘든 일을 털어놓으면?',
        options: [
            { text: '"그래서 원인이 뭐야? 해결책을 찾아보자." (T성향)', weights: { 'ml-eng': 4, 'data-sci': 3, 'research': 3 } },
            { text: '"많이 힘들었겠다.. 괜찮아?" (F성향)', weights: { 'pm': 5, 'prompt-eng': 4 } },
        ],
    },
    // 12. 서비스 개선
    {
        id: 12,
        question: '자주 쓰는 앱에서 불편한 점을 발견하면?',
        options: [
            { text: '"이런 기능 넣으면 대박 나겠는데?" 상상해본다.', weights: { 'pm': 5, 'ai-app': 3 } },
            { text: '"서버가 느린가? 코드를 어떻게 짰길래.." 분석한다.', weights: { 'mlops': 4, 'ml-eng': 3 } },
            { text: '그냥 참고 쓴다.', weights: { 'data-sci': 2 } },
        ],
    },
    // 13. 완벽주의
    {
        id: 13,
        question: '시험 공부를 할 때?',
        options: [
            { text: '기초부터 차근차근 원리를 이해해야 한다.', weights: { 'research': 5, 'data-sci': 4 } },
            { text: '중요한 기출문제 위주로 빠르게 훑는다.', weights: { 'ai-app': 4, 'pm': 3 } },
            { text: '나만의 요약 노트를 아주 예쁘게 만든다.', weights: { 'prompt-eng': 3, 'data-eng': 2 } },
        ],
    },
    // 14. 리더십
    {
        id: 14,
        question: '모임에서 장소를 정해야 할 때?',
        options: [
            { text: '맛집, 위치, 가격 비교해서 엑셀로 정리해 공유한다.', weights: { 'data-eng': 5, 'data-sci': 3, 'pm': 3 } },
            { text: '"어디 가고 싶어?" 의견을 듣고 결정한다.', weights: { 'pm': 5, 'ml-eng': 3 } },
            { text: '"그냥 아무 데나 가자." (따라가는 편)', weights: { 'research': 2, 'mlops': 2 } },
        ],
    },
    // 15. 창작 욕구
    {
        id: 15,
        question: '블로그나 SNS를 한다면?',
        options: [
            { text: '사람들이 좋아할 만한 꿀팁 정보를 올린다.', weights: { 'ai-app': 4, 'pm': 4 } },
            { text: '내가 공부한 내용을 기록용으로 정리한다.', weights: { 'research': 4, 'data-sci': 3, 'ml-eng': 3 } },
            { text: '감성적인 사진과 짧은 글을 올린다.', weights: { 'prompt-eng': 5 } },
        ],
    },
    // 16. 문제 직면
    {
        id: 16,
        question: '꽉 막힌 도로, 운전 중에 차가 안 움직인다면?',
        options: [
            { text: '내가 아는 지름길로 빠져나간다.', weights: { 'ai-app': 4, 'ml-eng': 3 } },
            { text: '네비게이션이 알려주는 도착 시간을 믿고 기다린다.', weights: { 'data-sci': 3, 'research': 2 } },
            { text: '동승자와 수다를 떨며 시간을 보낸다.', weights: { 'pm': 4, 'prompt-eng': 3 } },
        ],
    },
    // 17. 선호하는 게임
    {
        id: 17,
        question: '게임을 한다면 어떤 장르?',
        options: [
            { text: '심시티나 문명 같은 건설/경영 시뮬레이션', weights: { 'data-eng': 4, 'mlops': 4, 'pm': 3 } },
            { text: '롤(LoL)이나 오버워치 같은 팀플레이 경쟁', weights: { 'pm': 3, 'ml-eng': 3 } },
            { text: '혼자서 스토리를 즐기는 RPG', weights: { 'research': 3, 'prompt-eng': 3, 'ai-app': 2 } },
        ],
    },
    // 18. 변화 적응
    {
        id: 18,
        question: '자주 가던 식당 메뉴가 싹 바뀌었다면?',
        options: [
            { text: '새로운 메뉴를 도전해본다.', weights: { 'ai-app': 5, 'prompt-eng': 4 } },
            { text: '맛없으면 어쩌지.. 리뷰부터 확인한다.', weights: { 'data-sci': 4, 'research': 3 } },
            { text: '사장님께 왜 바뀌었는지 물어본다.', weights: { 'pm': 4 } },
        ],
    },
    // 19. 설명 능력
    {
        id: 19,
        question: '어려운 개념을 남에게 설명해야 한다면?',
        options: [
            { text: '예시를 들어서 아주 쉽게 비유한다.', weights: { 'pm': 5, 'prompt-eng': 5, 'ai-app': 3 } },
            { text: '정확한 용어와 정의를 사용해 설명한다.', weights: { 'research': 5, 'mlops': 3, 'data-eng': 3 } },
        ],
    },
    // 20. 최종 목표 (단순화)
    {
        id: 20,
        question: '나중에 어떤 사람으로 불리고 싶나요?',
        options: [
            { text: '"저 사람한테 맡기면 무조건 해결돼" (해결사)', weights: { 'ml-eng': 5, 'mlops': 4 } },
            { text: '"진짜 창의적이고 아이디어가 좋아" (크리에이터)', weights: { 'ai-app': 5, 'prompt-eng': 5 } },
            { text: '"아는 게 정말 많고 깊이가 있어" (전문가)', weights: { 'research': 5, 'data-sci': 5 } },
            { text: '"일 처리가 깔끔하고 정리를 잘해" (매니저)', weights: { 'pm': 5, 'data-eng': 4 } },
        ],
    },
    // 21. 하드웨어/로봇 (New for Physical AI)
    {
        id: 21,
        question: '개발하고 싶은 대상이 무엇인가요?',
        options: [
            { text: '모니터 속의 소프트웨어 (웹, 앱, 서버)', weights: { 'ai-app': 4, 'pm': 2, 'data-sci': 2 } },
            { text: '현실 세계에서 움직이는 기계 (로봇, 드론, 자동차)', weights: { 'physical-ai': 5, 'research': 2 } },
            { text: '데이터 그 자체 (숫자와 통계)', weights: { 'data-sci': 4, 'data-eng': 3 } },
            { text: '거대하고 복잡한 시스템 인프라', weights: { 'mlops': 4, 'data-eng': 4 } },
        ],
    },
];
