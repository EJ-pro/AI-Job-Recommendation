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
        description: 'LLM이 최적의 답변을 내놓도록 프롬프트를 설계/최적화하고, 모델의 한계를 극복합니다.',
        long_description: 'Prompt Engineer는 AI 모델과 소통하는 "통역사"입니다. 모델이 의도한 대로 정확하고 일관된 답변을 하도록 지시어(Prompt)를 설계하고, 다양한 케이스에 대해 실험하며 최적의 효율을 찾아냅니다. 최근에는 단순한 문장 작성을 넘어, 데이터를 평가하고 자동화하는 엔지니어링 영역으로 확장되고 있습니다.',
        salary_range: '초봉 3,500 ~ 5,000만원',
        difficulty: 'Medium',
        demand: 'Medium',
        responsibilities: [
            'System Prompt 설계 및 최적화',
            'Few-shot 예제 데이터셋 구축',
            'LLM 답변 품질 평가(Evaluation) 및 개선',
            '비용 및 응답 속도 최적화'
        ],
        tech_stack: [
            { category: 'Concepts', skills: ['CoT (Chain of Thought)', 'ReAct', 'Zero-shot/Few-shot'] },
            { category: 'Tools', skills: ['OpenAI Playground', 'Anthropic Console', 'LangSmith'] },
            { category: 'Scripting', skills: ['Python (Basic)', 'Jupyter Notebook'] }
        ],
        tags: ['Logical', 'Language', 'Creative'],
        focus_areas: [
            'Advanced Prompting (CoT, ReAct)',
            'LLM 평가 방법론 (Evals)',
            '기초 Python 스크립팅'
        ],
        roadmap: [
            {
                step: 'Phase 1: 프롬프트 기초',
                title: 'LLM의 작동 원리 이해',
                description: '모델이 텍스트를 생성하는 원리를 이해하고 기본적인 지시 기법을 익힙니다.',
                topics: ['Transformer 개요', 'Temperature & Top P', 'Role Play', 'Zero/One/Few-shot'],
                resources: [
                    { name: 'Prompt Engineering Guide', url: 'https://www.promptingguide.ai/' }
                ]
            },
            {
                step: 'Phase 2: 고급 기법',
                title: '복잡한 문제 해결하기',
                description: 'Chain of Thought 등 논리적 추론을 유도하는 고급 기법을 적용합니다.',
                topics: ['CoT', 'Tree of Thoughts', 'Self-Consistency', 'Prompt Chaining'],
                resources: [
                    { name: 'Anthropic Prompt Library', url: 'https://docs.anthropic.com/claude/prompt-library' }
                ]
            },
            {
                step: 'Phase 3: 엔지니어링 & 자동화',
                title: '프롬프트 최적화 자동화',
                description: 'Python 스크립트를 통해 프롬프트 테스트를 자동화하고 성능을 정량적으로 평가합니다.',
                topics: ['LLM Evaluation', 'OpenAI Evals', 'DSPy (Declarative Programming)', 'A/B Testing'],
                resources: [
                    { name: 'HPC AI Tech Blog', url: 'https://medium.com/@hpcai' }
                ]
            }
        ],
        faq: [
            { question: '개발 지식이 없어도 되나요?', answer: '초기에는 괜찮지만, 전문적인 커리어를 위해서는 파이썬을 활용한 데이터 처리나 자동화 스크립트 작성 능력은 필수입니다.' },
            { question: '미래에도 유망한가요?', answer: '모델이 똑똑해지면서 단순 프롬프팅은 줄겠지만, 복잡한 시스템을 지휘하고 평가하는 "AI 오케스트레이션" 능력은 더 중요해질 것입니다.' }
        ]
    },
    {
        id: 'mlops',
        title: 'MLOps Engineer',
        description: '머신러닝 모델의 학습부터 배포, 모니터링까지의 전체 라이프사이클을 자동화하고 관리합니다.',
        long_description: 'MLOps Engineer는 "Machine Learning"과 "Operations(운영)"의 합성어로, AI 모델을 연구실에서 꺼내 실제 서비스 환경에서 안정적으로 돌아가게 만드는 핵심 인프라 전문가입니다. DevOps 문화를 ML에 적용하여, 모델 학습-배포-모니터링의 과정을 자동화(CI/CD/CT)합니다.',
        salary_range: '초봉 4,500 ~ 6,000만원',
        difficulty: 'Hard',
        demand: 'High',
        responsibilities: [
            'ML 파이프라인(학습/전처리/배포) 자동화 구축',
            'Kubernetes 기반의 모델 서빙 인프라 관리',
            '모델 성능 및 리소스 모니터링 시스템 구축',
            '클라우드(AWS/GCP) 비용 최적화'
        ],
        tech_stack: [
            { category: 'Cloud', skills: ['AWS', 'GCP', 'Azure'] },
            { category: 'Container', skills: ['Docker', 'Kubernetes', 'Helm'] },
            { category: 'CI/CD', skills: ['GitHub Actions', 'Jenkins', 'ArgoCD'] },
            { category: 'MLOps Tools', skills: ['MLflow', 'Kubeflow', 'Airflow', 'Prometheus'] }
        ],
        tags: ['Infrastructure', 'DevOps', 'Stability'],
        focus_areas: [
            'Docker & Kubernetes',
            'Cloud Platform (AWS/GCP)',
            'CI/CD for ML (GitHub Actions)'
        ],
        roadmap: [
            {
                step: 'Phase 1: 컨테이너 & 클라우드',
                title: 'Docker와 AWS 기초',
                description: '애플리케이션을 격리된 환경(컨테이너)으로 만들고 클라우드 서버에 배포합니다.',
                topics: ['DockerFile 작성', 'EC2/Lambda 배포', 'Linux 터미널 명령어', 'Networking 기초'],
                resources: [
                    { name: 'Docker for Beginners', url: 'https://docker-curriculum.com/' }
                ]
            },
            {
                step: 'Phase 2: CI/CD 파이프라인',
                title: '자동화 시스템 구축',
                description: '코드가 변경되면 자동으로 테스트하고 배포하는 파이프라인을 만듭니다.',
                topics: ['GitHub Actions', 'Unit Testing', 'Automated Deployment', 'Model Registry (MLflow)'],
                resources: [
                    { name: 'MLOps Zoomcamp', url: 'https://github.com/DataTalksClub/mlops-zoomcamp' }
                ]
            },
            {
                step: 'Phase 3: 오케스트레이션 & 모니터링',
                title: 'Kubernetes & Serving',
                description: '대규모 트래픽을 처리하는 클러스터를 운영하고 모델 상태를 감시합니다.',
                topics: ['K8s Pods/Services', 'Model Serving (Triton/TorchServe)', 'Grafana/Prometheus', 'GPU Resource Management'],
                resources: [
                    { name: 'Kubernetes Tutorials', url: 'https://kubernetes.io/docs/tutorials/' }
                ]
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
        description: '데이터의 수집, 저장, 처리를 위한 견고한 파이프라인을 구축하여 모델 학습을 지원합니다.',
        long_description: 'Data Engineer는 데이터의 "배관공"입니다. 다양한 곳에 흩어진 데이터를 수집(Extract)하고, 사용하기 좋게 가공(Transform)하여, 저장소(Load)에 적재하는 ETL 파이프라인을 책임집니다. 데이터 과학자나 분석가가 데이터를 분석할 수 있도록 깨끗한 데이터를 안정적으로 공급하는 것이 목표입니다.',
        salary_range: '초봉 4,000 ~ 5,500만원',
        difficulty: 'Medium',
        demand: 'High',
        responsibilities: [
            'ETL/ELT 파이프라인 설계 및 운영',
            'Data Warehouse / Data Lake 구축 및 관리',
            '대용량 데이터 분산 처리 (Spark 등)',
            '데이터 품질(Quality) 및 정합성 관리'
        ],
        tech_stack: [
            { category: 'Language', skills: ['Python', 'SQL', 'Scala', 'Java'] },
            { category: 'Big Data', skills: ['Apache Spark', 'Kafka', 'Hadoop'] },
            { category: 'Workflow', skills: ['Apache Airflow', 'Prefect', 'dbt'] },
            { category: 'Storage', skills: ['Snowflake', 'BigQuery', 'Redshift', 'S3'] }
        ],
        tags: ['BigData', 'Pipeline', 'Spark/Kafka'],
        focus_areas: [
            'SQL & Database Design',
            'Python ETL (Airflow/Prefect)',
            'Cloud Data Warehouse (BigQuery 등)'
        ],
        roadmap: [
            {
                step: 'Phase 1: 데이터 베이스 기초',
                title: 'SQL Master & Python Data',
                description: '데이터를 저장하고 조회하는 가장 기본적인 언어를 완벽하게 익힙니다.',
                topics: ['Advanced SQL (Window Functions)', 'Data Modeling (Star/Snowflake Schema)', 'Pandas Data Cleaning'],
                resources: [
                    { name: 'Mode SQL Tutorial', url: 'https://mode.com/sql-tutorial/' }
                ]
            },
            {
                step: 'Phase 2: 파이프라인 구축',
                title: 'Airflow & ETL',
                description: '데이터 이동을 스케줄링하고 자동화하는 워크플로우를 만듭니다.',
                topics: ['DAG 작성', 'Cron Expression', 'Web Scraping -> DB 적재', 'API Data Fetching'],
                resources: [
                    { name: 'Apache Airflow Docs', url: 'https://airflow.apache.org/' }
                ]
            },
            {
                step: 'Phase 3: 빅데이터 & 클라우드',
                title: 'Distributed Processing',
                description: '메모리에 다 들어가지 않는 거대한 데이터를 다루는 기술을 배웁니다.',
                topics: ['Spark Architecture', 'AWS EMR / Glue', 'Streaming Data (Kafka)', 'Data Governance'],
                resources: [
                    { name: 'Data Engineering Cookbook', url: 'https://github.com/andkret/Cookbook' }
                ]
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
        description: '복잡한 데이터에서 비즈니스 인사이트를 도출하고, 통계 및 머신러닝 알고리즘을 적용합니다.',
        long_description: 'Data Scientist는 데이터를 통해 "가치"를 발견하는 탐험가입니다. 수집된 데이터에서 패턴을 찾고(EDA), 통계적 기법으로 가설을 검증하며, 예측 모델을 만들어 비즈니스 의사결정을 돕습니다. 기술적인 능력(코딩)뿐만 아니라 비즈니스 도메인에 대한 깊은 이해와 스토리텔링 능력이 필수적입니다.',
        salary_range: '초봉 4,000 ~ 5,500만원',
        difficulty: 'Hard',
        demand: 'Medium',
        responsibilities: [
            '데이터 탐색적 분석(EDA) 및 시각화',
            '예측 모델링 및 머신러닝 알고리즘 적용',
            'A/B 테스트 설계 및 결과 분석',
            '비즈니스 인사이트 도출 및 리포팅'
        ],
        tech_stack: [
            { category: 'Language', skills: ['Python', 'R', 'SQL'] },
            { category: 'Analysis', skills: ['Pandas', 'NumPy', 'Scipy'] },
            { category: 'ML', skills: ['Scikit-learn', 'XGBoost', 'LightGBM'] },
            { category: 'Visualization', skills: ['Matplotlib', 'Seaborn', 'Tableau', 'PowerBI'] }
        ],
        tags: ['Analysis', 'Statistics', 'Math'],
        focus_areas: [
            'Python 데이터 분석 (Pandas/Scikit-learn)',
            '통계학 기초 및 가설 검정',
            '데이터 시각화/스토리텔링'
        ],
        roadmap: [
            {
                step: 'Phase 1: 데이터 분석 기초',
                title: 'Data Analysis & Visualization',
                description: '데이터를 요리조리 뜯어보고 시각적으로 표현하는 능력을 기릅니다.',
                topics: ['Python Basics', 'Pandas DataFrame', 'Charts & Graphs', 'Descriptive Statistics'],
                resources: [
                    { name: 'Kaggle Learn', url: 'https://www.kaggle.com/learn' }
                ]
            },
            {
                step: 'Phase 2: 머신러닝 & 통계',
                title: 'Modeling & Inference',
                description: '데이터로 미래를 예측하거나 숨겨진 관계를 증명합니다.',
                topics: ['Regression/Classification', 'Hypothesis Testing', 'Feature Engineering', 'Model Evaluation Metrics'],
                resources: [
                    { name: 'StatQuest with Josh Starmer', url: 'https://www.youtube.com/c/joshstarmer' }
                ]
            },
            {
                step: 'Phase 3: 실전 문제 해결',
                title: 'Business Project',
                description: '실제 비즈니스 시나리오에서 문제를 정의하고 해결책을 제안합니다.',
                topics: ['Problem Definition', 'Dashboard Building', 'Presentation Skills', 'Domain Knowledge'],
                resources: [
                    { name: 'Towards Data Science', url: 'https://towardsdatascience.com/' }
                ]
            }
        ],
        faq: [
            { question: '석/박사 학위가 필수인가요?', answer: '필수는 아니지만, 통계적 지식의 깊이가 요구되므로 우대하는 경향이 있습니다. 포트폴리오로 실력을 증명하면 학력은 극복 가능합니다.' },
            { question: '취업이 어렵나요?', answer: '단순 분석가는 포화 상태지만, 비즈니스 감각과 엔지니어링 능력을 겸비한 "Full-stack Data Scientist"는 여전히 수요가 많습니다.' }
        ]
    },
    {
        id: 'research',
        title: 'AI Research Scientist',
        description: '새로운 알고리즘을 연구하거나 최신 논문을 구현하여 기술의 한계를 넓힙니다.',
        long_description: 'AI Research Scientist는 "미래의 기술"을 만드는 과학자입니다. 기존 모델의 성능을 뛰어넘는 새로운 아키텍처를 고안하거나, 아직 해결되지 않은 난제들을 딥러닝 기술로 풀어냅니다. 최신 논문을 끊임없이 읽고 구현하며, 학회에 논문을 발표하거나 기업의 핵심 원천 기술(Core Tech)을 개발합니다.',
        salary_range: '초봉 5,000 ~ 8,000만원 (학위별 상이)',
        difficulty: 'Extreme',
        demand: 'Medium',
        responsibilities: [
            '최신 AI 논문 리서치 및 구현 (SoTA 추구)',
            '신규 모델 아키텍처 설계 및 실험',
            '모델 경량화, 최적화 연구',
            '특허 출원 및 논문 작성'
        ],
        tech_stack: [
            { category: 'Framework', skills: ['PyTorch', 'TensorFlow', 'JAX'] },
            { category: 'Theory', skills: ['Linear Algebra', 'Probability', 'Calculus', 'Optimization'] },
            { category: 'Domain', skills: ['CV (Computer Vision)', 'NLP (Natural Language Processing)', 'RL (Reinforcement Learning)'] },
            { category: 'Tools', skills: ['LaTeX', 'Weights & Biases', 'Linux'] }
        ],
        tags: ['Research', 'DeepLearning', 'Academic'],
        focus_areas: [
            'Deep Learning Theory (PyTorch)',
            '최신 논문 리딩 및 구현 능력',
            '수학적 기초 (선형대수/확률)'
        ],
        roadmap: [
            {
                step: 'Phase 1: 딥러닝 이론',
                title: 'Deep Learning Foundations',
                description: '신경망의 수학적 원리를 바닥부터 이해합니다.',
                topics: ['Backpropagation', 'Gradient Descent', 'CNN/RNN Architecture', 'Pytorch Basics'],
                resources: [
                    { name: 'CS231n (Stanford)', url: 'http://cs231n.stanford.edu/' },
                    { name: 'Deep Learning Book', url: 'https://www.deeplearningbook.org/' }
                ]
            },
            {
                step: 'Phase 2: 논문 구현',
                title: 'Paper Reproduction',
                description: '유명한 논문을 읽고 코드로 똑같이 구현하며 디테일을 익힙니다.',
                topics: ['Reading Papers', 'Model Debugging', 'Training Tricks', 'Transformer Deep Dive'],
                resources: [
                    { name: 'Papers With Code', url: 'https://paperswithcode.com/' }
                ]
            },
            {
                step: 'Phase 3: 연구 수행',
                title: 'Novel Research',
                description: '자신만의 아이디어를 제안하고 실험을 통해 증명합니다.',
                topics: ['Experiment Design', 'Ablation Study', 'Writing Papers', 'Conference Submission'],
                resources: [
                    { name: 'ArXiv', url: 'https://arxiv.org/' }
                ]
            }
        ],
        faq: [
            { question: '박사 학위가 꼭 필요한가요?', answer: '대부분의 리서치 직군은 석사/박사 학위를 강력히 선호합니다. 학사라면 뛰어난 구현 능력이나 논문 실적을 보여줘야 합니다.' },
            { question: '수학을 얼마나 잘해야 하나요?', answer: '논문의 수식을 이해하고 코드로 옮길 수 있을 정도의 선형대수와 확률통계 지식은 필수입니다.' }
        ]
    },
    {
        id: 'pm',
        title: 'AI Product Manager',
        description: 'AI 기술을 활용한 제품을 기획하고, 개발팀과 비즈니스팀 사이의 가교 역할을 합니다.',
        long_description: 'AI Product Manager는 기술과 시장을 연결하는 "지휘자"입니다. 소비자가 원하는 것이 무엇인지 파악하고, 이를 AI 기술로 어떻게 해결할 수 있을지 기획합니다. 개발자, 디자이너, 비즈니스 팀과 끊임없이 소통하며 제품이 성공적으로 출시되고 운영되도록 관리합니다. 기술에 대한 이해도가 높을수록 좋은 PM이 될 수 있습니다.',
        salary_range: '초봉 3,500 ~ 5,500만원',
        difficulty: 'Medium',
        demand: 'High',
        responsibilities: [
            'AI 기반 서비스 기획 및 요구사항 명세(PRD) 작성',
            '제품 로드맵 수립 및 일정 관리',
            '데이터 기반 의사결정 및 성과 분석',
            '유관부서 커뮤니케이션 및 조율'
        ],
        tech_stack: [
            { category: 'Planning', skills: ['Figma', 'Jira', 'Confluence', 'Notion'] },
            { category: 'Data', skills: ['SQL', 'Google Analytics', 'Excel/Spreadsheet'] },
            { category: 'Tech Literacy', skills: ['API 이해', 'ML 개발 프로세스 이해', 'Prompting'] }
        ],
        tags: ['Communication', 'Planning', 'Business'],
        focus_areas: [
            'AI 기술 이해 (한계와 가능성)',
            '데이터 기반 의사결정 (SQL/GA)',
            '기획서/요구사항 명세서 작성'
        ],
        roadmap: [
            {
                step: 'Phase 1: 기획 역량',
                title: 'Service Planning Basics',
                description: '문제를 정의하고 해결책을 문서화하는 능력을 기릅니다.',
                topics: ['User Persona', 'User Journey Map', 'Wireframing', 'PRD 작성법'],
                resources: [
                    { name: 'Brunch (Planner Tips)', url: 'https://brunch.co.kr/' }
                ]
            },
            {
                step: 'Phase 2: AI 리터러시',
                title: 'Understanding AI',
                description: 'AI로 무엇이 가능하고 불가능한지 기술적 한계를 이해합니다.',
                topics: ['AI Terminology', 'Development Cycle', 'Cost & Latency', 'API Capabilities'],
                resources: [
                    { name: 'Google AI for Everyone', url: 'https://www.coursera.org/learn/ai-for-everyone' }
                ]
            },
            {
                step: 'Phase 3: 실전 매니지먼트',
                title: 'Product Launching',
                description: '실제 제품을 만들어보고 데이터를 보며 개선합니다.',
                topics: ['Agile/Scrum', 'Data Driven Decision', 'A/B Testing', 'Growth Hacking'],
                resources: [
                    { name: 'Lenny\'s Newsletter', url: 'https://www.lennysnewsletter.com/' }
                ]
            }
        ],
        faq: [
            { question: '개발을 할 줄 알아야 하나요?', answer: '직접 코딩은 안 해도 되지만, 개발자와 대화가 통할 정도의 기술 이해도(API, DB, 서버 등)는 반드시 필요합니다.' },
            { question: '어떤 전공이 유리한가요?', answer: '전공 무관합니다. 인문계열이라도 논리적 사고력과 커뮤니케이션 능력이 뛰어나면 충분합니다.' }
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
