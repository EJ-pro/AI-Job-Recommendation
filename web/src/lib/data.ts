export interface JobRole {
    id: string;
    title: string;
    description: string;
    tasks: string[];
    tags: string[];
    focus_areas: string[]; // 3개월 집중 공략 분야 (Top 3)
    roadmap: { step: string; action: string }[]; // 3단계 학습 로드맵
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
        tasks: ['LangChain/LLM 연동', 'AI 서비스 백엔드 개발', 'RAG(검색 증강) 시스템 구축'],
        tags: ['Service', 'API', 'Fast-Paced'],
        focus_areas: [
            'LangChain/LlamaIndex 프레임워크 마스터',
            'Python 백엔드 (FastAPI/Streamlit)',
            'Prompt Engineering 기초'
        ],
        roadmap: [
            { step: '1개월차', action: 'OpenAI API와 LangChain 튜토리얼을 따라하며 챗봇 3개 만들기 (문서요약, Q&A 등)' },
            { step: '2개월차', action: 'Streamlit이나 FastAPI를 사용하여 웹 서비스 형태로 배포하고, Vector DB(Pinecone 등) 연동해보기' },
            { step: '3개월차', action: '나만의 AI 서비스(기획+개발+배포) 포트폴리오 1개 완성하기 (RAG 포함)' }
        ]
    },
    {
        id: 'prompt-eng',
        title: 'Prompt Engineer',
        description: 'LLM이 최적의 답변을 내놓도록 프롬프트를 설계/최적화하고, 모델의 한계를 극복합니다.',
        tasks: ['프롬프트 최적화 및 평가', 'Few-shot Learning 설계', '모델 환각(Hallucination) 제어'],
        tags: ['Logical', 'Language', 'Creative'],
        focus_areas: [
            'Advanced Prompting (CoT, ReAct)',
            'LLM 평가 방법론 (Evals)',
            '기초 Python 스크립팅'
        ],
        roadmap: [
            { step: '1개월차', action: '기본적인 프롬프트 기법(Zero/Few-shot, CoT)을 다양한 모델(GPT, Claude)에서 실험하고 기록하기' },
            { step: '2개월차', action: '복잡한 태스크를 해결하는 프롬프트 체인을 설계하고, 자동화 스크립트 작성해보기' },
            { step: '3개월차', action: '특정 도메인(법률, 의료 등)에 특화된 프롬프트 셋과 평가 리포트를 포트폴리오로 정리하기' }
        ]
    },
    {
        id: 'mlops',
        title: 'MLOps Engineer',
        description: '머신러닝 모델의 학습부터 배포, 모니터링까지의 전체 라이프사이클을 자동화하고 관리합니다.',
        tasks: ['모델 배포 및 서빙', '클라우드 인프라(AWS/GCP) 관리', '자동화 파이프라인 구축'],
        tags: ['Infrastructure', 'DevOps', 'Stability'],
        focus_areas: [
            'Docker & Kubernetes',
            'Cloud Platform (AWS/GCP)',
            'CI/CD for ML (GitHub Actions)'
        ],
        roadmap: [
            { step: '1개월차', action: 'Python으로 만든 간단한 모델을 Docker Container로 패키징하고 AWS EC2에 띄워보기' },
            { step: '2개월차', action: 'GitHub Actions를 이용해 모델 학습부터 배포까지 자동화되는 CI/CD 파이프라인 구축하기' },
            { step: '3개월차', action: 'Kubernetes 혹은 Serverless(Lambda) 환경에서 모델 서빙 아키텍처 구현 및 트래픽 테스트' }
        ]
    },
    {
        id: 'data-eng',
        title: 'Data Engineer',
        description: '데이터의 수집, 저장, 처리를 위한 견고한 파이프라인을 구축하여 모델 학습을 지원합니다.',
        tasks: ['ETL 파이프라인 구축', '데이터 웨어하우스/레이크 관리', '대용량 데이터 분산 처리'],
        tags: ['BigData', 'Pipeline', 'Spark/Kafka'],
        focus_areas: [
            'SQL & Database Design',
            'Python ETL (Airflow/Prefect)',
            'Cloud Data Warehouse (BigQuery 등)'
        ],
        roadmap: [
            { step: '1개월차', action: 'SQL 심화 문법 완성 및 Python Pandas로 데이터 정제 스크립트 능숙하게 다루기' },
            { step: '2개월차', action: 'Airflow를 설치하여 주기적으로 데이터를 수집/가공하는 ETL 파이프라인 만들어보기' },
            { step: '3개월차', action: '클라우드(AWS S3, BigQuery)를 연동하여 대시보드까지 이어지는 전체 데이터 흐름 완성하기' }
        ]
    },
    {
        id: 'data-sci',
        title: 'Data Scientist',
        description: '복잡한 데이터에서 비즈니스 인사이트를 도출하고, 통계 및 머신러닝 알고리즘을 적용합니다.',
        tasks: ['데이터 탐색적 분석(EDA)', '시각화 및 대시보드 제작', '통계적 가설 검정'],
        tags: ['Analysis', 'Statistics', 'Math'],
        focus_areas: [
            'Python 데이터 분석 (Pandas/Scikit-learn)',
            '통계학 기초 및 가설 검정',
            '데이터 시각화/스토리텔링'
        ],
        roadmap: [
            { step: '1개월차', action: 'Kaggle의 유명한 데이터셋 3개를 골라 EDA(탐색적 데이터 분석) 리포트 작성해보기' },
            { step: '2개월차', action: 'Scikit-learn을 활용해 분류/회귀 모델을 만들고, 성능 지표(Accuracy, F1 등) 해석 능력 기르기' },
            { step: '3개월차', action: '자신만의 가설을 세우고 데이터를 통해 검증하여 비즈니스 제안까지 포함된 분석 프로젝트 완성' }
        ]
    },
    {
        id: 'research',
        title: 'AI Research Scientist',
        description: '새로운 알고리즘을 연구하거나 최신 논문을 구현하여 기술의 한계를 넓힙니다.',
        tasks: ['논문 리서치 및 구현', '모델 아키텍처 설계', '모델 경량화 및 최적화'],
        tags: ['Research', 'DeepLearning', 'Academic'],
        focus_areas: [
            'Deep Learning Theory (PyTorch)',
            '최신 논문 리딩 및 구현 능력',
            '수학적 기초 (선형대수/확률)'
        ],
        roadmap: [
            { step: '1개월차', action: 'PyTorch 공식 튜토리얼 완독 및 CNN/RNN/Transformer 기본 구조 바닥부터 코딩해보기' },
            { step: '2개월차', action: '관심 분야(CV/NLP)의 Top-tier 학회 최신 논문 1편을 선정하여 코드로 구현(Reproduction)하기' },
            { step: '3개월차', action: '구현한 모델을 학습시켜 벤치마크 성능을 측정하고, 개선 아이디어 실험해보기' }
        ]
    },
    {
        id: 'pm',
        title: 'AI Product Manager',
        description: 'AI 기술을 활용한 제품을 기획하고, 개발팀과 비즈니스팀 사이의 가교 역할을 합니다.',
        tasks: ['사용자 요구사항 정의', '제품 로드맵 설계', 'AI 서비스 기획'],
        tags: ['Communication', 'Planning', 'Business'],
        focus_areas: [
            'AI 기술 이해 (한계와 가능성)',
            '데이터 기반 의사결정 (SQL/GA)',
            '기획서/요구사항 명세서 작성'
        ],
        roadmap: [
            { step: '1개월차', action: '성공한 AI 서비스 5개를 분석(Reverse Engineering)하여 기능 명세서와 UX 흐름도 역기획해보기' },
            { step: '2개월차', action: '간단한 노코드 툴이나 API를 활용해 MVP(최소기능제품)를 직접 만들어보고 사용자 반응 보기' },
            { step: '3개월차', action: '해결하려는 문제 정의부터 솔루션, 지표 설정까지 포함된 완벽한 서비스 기획 포트폴리오 작성' }
        ]
    },
    {
        id: 'ml-eng',
        title: 'Machine Learning Engineer',
        description: '머신러닝 알고리즘을 활용해 데이터를 학습시키고, 실제 서비스에 적용하여 가치를 창출하는 모델을 개발합니다.',
        tasks: ['머신러닝 모델 설계 및 개발', '데이터 전처리 및 피처 엔지니어링', '모델 성능 평가 및 최적화'],
        tags: ['Modeling', 'Engineering', 'AI'],
        focus_areas: [
            'Deep Learning (PyTorch/TensorFlow)',
            'ML Algorithms & Mathematics',
            'Model Serving & MLOps Basics'
        ],
        roadmap: [
            { step: '1개월차', action: '기초 ML/DL 이론(회귀, 분류, 신경망)을 학습하고 Scikit-learn으로 간단한 모델 구현하기' },
            { step: '2개월차', action: 'Kaggle 등의 데이터셋을 활용해 탐색적 분석(EDA)부터 모델 학습, 성능 개선까지 수행하기' },
            { step: '3개월차', action: '학습된 모델을 웹 서비스(Streamlit/FastAPI)에 연동하고 실제 사용자에게 배포하는 경험 쌓기' }
        ]
    },
    {
        id: 'physical-ai',
        title: 'Physical AI Engineer',
        description: '현실 세계의 로봇이나 장치를 제어하는 AI 모델을 개발하고, 센서 데이터를 처리하여 물리적인 상호작용을 구현합니다.',
        tasks: ['로봇/드론 자율주행 알고리즘 개발', '센서 데이터(LiDAR, Camera) 퓨전 및 처리', '임베디드 AI 모델 최적화 (Edge AI)'],
        tags: ['Robotics', 'Embedded', 'Hardware'],
        focus_areas: [
            'ROS (Robot Operating System) & Gazebo Sim',
            'Computer Vision (SLAM, Object Detection)',
            'Embedded System (NVIDIA Jetson, Raspberry Pi)'
        ],
        roadmap: [
            { step: '1개월차', action: 'Python/C++로 기본 로봇 제어(모터/센서) 및 ROS 2 기초 학습하기' },
            { step: '2개월차', action: '카메라/LiDAR 센서 데이터를 받아 장애물을 피하는 자율주행 봇 시뮬레이션 구현하기' },
            { step: '3개월차', action: '실제 임베디드 보드(Jetson 등)에 경량화된 AI 모델을 탑재하여 미션을 수행하는 프로젝트 완성하기' }
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
