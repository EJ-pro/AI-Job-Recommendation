export interface JobRole {
    id: string;
    title: string;
    description: string;
    tasks: string[];
    tags: string[];
}

export interface Question {
    id: number;
    question: string;
    options: {
        text: string;
        weights: Partial<Record<string, number>>; // job_id -> score
    }[];
}

export const JOBS: JobRole[] = [
    {
        id: 'ai-app',
        title: 'AI Application Engineer',
        description: '기존 AI 모델(GPT, Stable Diffusion 등)을 API로 연동하여 바로 서비스를 만드는 역할입니다.',
        tasks: ['LangChain/LLM 연동', 'AI 서비스 백엔드 개발', 'Prompt Engineering'],
        tags: ['Service-Oriented', 'API', 'Fast-Paced'],
    },
    {
        id: 'applied-ai',
        title: 'Applied AI Engineer',
        description: '최신 논문이나 모델을 실제 비즈니스 문제에 맞게 튜닝하고 적용하는 역할입니다.',
        tasks: ['RAG 시스템 구축', '모델 Fine-tuning', 'MLOps 파이프라인 구성'],
        tags: ['Model-Tuning', 'Engineering', 'Problem-Solving'],
    },
    {
        id: 'backend',
        title: 'AI Backend Engineer',
        description: '대규모 트래픽을 처리하고 AI 모델 서빙 인프라를 구축하는 백엔드 개발자입니다.',
        tasks: ['모델 서빙 API 개발', '대용량 데이터 처리', '클라우드 인프라(AWS/GCP)'],
        tags: ['Infrastructure', 'Scalability', 'System Design'],
    },
    {
        id: 'frontend',
        title: 'AI Frontend Engineer',
        description: 'AI 기술을 활용한 웹/앱 인터페이스를 구현하고 사용자 경험을 최적화합니다.',
        tasks: ['AI 챗봇 UI 구현', '데이터 시각화', '인터랙티브 웹 개발'],
        tags: ['UI/UX', 'React/Next.js', 'Visualization'],
    },
    {
        id: 'data-analyst',
        title: 'Data Analyst',
        description: '데이터를 분석하여 비즈니스 인사이트를 도출하고 의사결정을 지원합니다.',
        tasks: ['SQL 데이터 추출', '대시보드 시각화', '통계 분석'],
        tags: ['Statistics', 'Insight', 'Business'],
    },
    {
        id: 'product',
        title: 'AI Product Engineer',
        description: '기술과 비즈니스를 연결하며 AI 제품의 기획부터 개발까지 주도하는 역할입니다.',
        tasks: ['제품 기획 및 설계', '프로토타이핑', '사용자 요구사항 분석'],
        tags: ['Planning', 'Full-stack', 'Communication'],
    },
];

export const QUESTIONS: Question[] = [
    {
        id: 1,
        question: '개발 공부를 할 때 더 흥미로운 것은 무엇인가요?',
        options: [
            {
                text: '눈에 보이는 화면을 만들고 사용자와 상호작용하는 것',
                weights: { 'frontend': 5, 'product': 3, 'ai-app': 2 }
            },
            {
                text: '눈에 보이지 않지만 시스템이 안정적으로 돌아가게 로직을 짜는 것',
                weights: { 'backend': 5, 'applied-ai': 3, 'data-analyst': 2 }
            },
        ],
    },
    {
        id: 2,
        question: 'AI 기술을 어떻게 활용하고 싶나요?',
        options: [
            {
                text: '성능 좋은 최신 모델(LLM)을 가져와서 멋진 서비스를 빠르게 만들고 싶다.',
                weights: { 'ai-app': 5, 'product': 4, 'frontend': 2 }
            },
            {
                text: '모델이 어떻게 작동하는지 이해하고, 내 데이터에 맞게 튜닝해보고 싶다.',
                weights: { 'applied-ai': 5, 'data-analyst': 3 }
            },
        ],
    },
    {
        id: 3,
        question: '데이터를 다루는 것에 대해 어떻게 생각하나요?',
        options: [
            {
                text: '데이터 속에서 패턴을 찾고 그래프로 시각화하는 것이 재밌다.',
                weights: { 'data-analyst': 5, 'applied-ai': 3, 'product': 2 }
            },
            {
                text: '데이터 자체보다는 데이터를 주고받는 API나 시스템 구조가 더 중요하다.',
                weights: { 'backend': 5, 'ai-app': 4, 'frontend': 2 }
            },
        ],
    },
    {
        id: 4,
        question: '프로젝트를 진행할 때 어떤 역할을 선호하나요?',
        options: [
            {
                text: '기획부터 개발까지 주도적으로 참여하고 전체 그림을 보는 것이 좋다.',
                weights: { 'product': 5, 'ai-app': 3 }
            },
            {
                text: '주어진 기술적 난제를 깊이 있게 파고들어 해결하는 것이 좋다.',
                weights: { 'backend': 4, 'applied-ai': 5, 'frontend': 3 }
            },
        ],
    },
    {
        id: 5,
        question: 'Python과 JavaScript 중 더 자신 있는 언어는?',
        options: [
            {
                text: 'Python (데이터 처리, 모델링 라이브러리 활용)',
                weights: { 'applied-ai': 4, 'data-analyst': 5, 'backend': 3 }
            },
            {
                text: 'JavaScript/TypeScript (웹 개발, 인터랙션)',
                weights: { 'frontend': 5, 'ai-app': 4, 'product': 3 }
            },
        ],
    },
    {
        id: 6,
        question: '가장 끌리는 키워드는?',
        options: [
            {
                text: '사용자 경험(UX), 디자인, 인터랙션',
                weights: { 'frontend': 5, 'product': 4 }
            },
            {
                text: '서버 성능, 클라우드, 아키텍처',
                weights: { 'backend': 5, 'applied-ai': 3 }
            },
            {
                text: '데이터 분석, 통계, 인사이트',
                weights: { 'data-analyst': 5, 'product': 2 }
            },
            {
                text: '생성형 AI, 프롬프트, 챗봇',
                weights: { 'ai-app': 5, 'applied-ai': 3 }
            },
        ],
    },
    {
        id: 7,
        question: '협업 시 어떤 칭찬을 듣고 싶나요?',
        options: [
            {
                text: '센스 있게 기능을 잘 구현하고 정리도 깔끔하다.',
                weights: { 'frontend': 3, 'ai-app': 3, 'product': 3 }
            },
            {
                text: '어려운 기술적 문제를 끈기 있게 해결했다.',
                weights: { 'backend': 4, 'applied-ai': 5 }
            },
        ],
    },
    // 6~8 questions requested. Implemented 7.
];
