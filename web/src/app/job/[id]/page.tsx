import { JOBS, TechStack, RoadmapStep } from '@/lib/data';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import Button from '@/components/Button';
import clsx from 'clsx';
import { JobReadinessTracker } from '@/components/JobReadinessTracker';
import {
    ArrowLeft,
    CheckCircle2,
    Map as MapIcon,
    Target,
    Briefcase,
    TrendingUp,
    DollarSign,
    BookOpen,
    HelpCircle,
    Layers,
    Code2,
    Database,
    Cpu,
    ExternalLink,
    Zap
} from 'lucide-react';

interface PageProps {
    params: {
        id: string;
    };
}

export function generateStaticParams() {
    return JOBS.map((job) => ({
        id: job.id,
    }));
}

export default async function JobDetailPage({ params }: PageProps) {
    const { id } = await params;
    const job = JOBS.find((j) => j.id === id);

    if (!job) {
        notFound();
    }

    return (
        <div className="min-h-screen bg-background text-foreground animate-in fade-in slide-in-from-bottom-4 duration-700">

            {/* 1. Header / Navigation */}
            <div className="sticky top-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
                <div className="max-w-5xl mx-auto px-4 md:px-8 py-3 flex items-center justify-between">
                    <Link href="/" className="group">
                        <Button variant="ghost" className="pl-0 hover:pl-2 transition-all gap-2 text-muted-foreground hover:text-foreground">
                            <ArrowLeft size={18} />
                            <span className="font-medium">다른 직무 보기</span>
                        </Button>
                    </Link>
                    <div className="text-sm font-semibold text-primary">
                        AI Career Guide
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-4 md:px-8 py-12 space-y-20">

                {/* 2. Hero Section */}
                <section className="space-y-8 text-center md:text-left">
                    <div className="space-y-4">
                        <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                            {job.tags.map(tag => (
                                <span key={tag} className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-semibold tracking-wide border border-primary/20">
                                    #{tag}
                                </span>
                            ))}
                        </div>
                        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-foreground leading-tight">
                            {job.title}
                            <span className="text-primary">.</span>
                        </h1>
                        <p className="text-xl md:text-2xl text-muted-foreground leading-relaxed max-w-3xl">
                            {job.description}
                        </p>
                        <div className="pt-4 max-w-3xl text-lg text-foreground/80 leading-relaxed bg-secondary/30 p-6 rounded-2xl border border-border">
                            {job.long_description}
                        </div>
                    </div>

                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <StatCard
                            icon={<DollarSign className="text-green-500" />}
                            label="평균 연봉 (신입)"
                            value={job.salary_range}
                        />
                        <StatCard
                            icon={<TrendingUp className="text-blue-500" />}
                            label="시장 수요"
                            value={job.demand}
                            highlight={job.demand === 'Very High' || job.demand === 'High'}
                        />
                        <StatCard
                            icon={<Zap className="text-amber-500" />}
                            label="난이도"
                            value={job.difficulty}
                        />
                    </div>
                </section>

                {/* 3. Responsibilities & Tech Stack */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                    {/* Responsibilities */}
                    <section className="space-y-6">
                        <h2 className="text-2xl font-bold flex items-center gap-2">
                            <Briefcase className="text-primary" />
                            주요 업무 (Key Responsibilities)
                        </h2>
                        <div className="bg-card border border-border rounded-2xl p-6 shadow-sm">
                            <ul className="space-y-4">
                                {job.responsibilities.map((resp, i) => (
                                    <li key={i} className="flex items-start gap-3">
                                        <CheckCircle2 className="text-primary mt-1 shrink-0" size={18} />
                                        <span className="text-foreground/90 leading-relaxed">{resp}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </section>

                    {/* Tech Stack */}
                    <section className="space-y-6">
                        <h2 className="text-2xl font-bold flex items-center gap-2">
                            <Layers className="text-primary" />
                            기술 스택 (Tech Stack)
                        </h2>
                        <div className="bg-card border border-border rounded-2xl p-6 shadow-sm space-y-6">
                            {job.tech_stack.map((stack, i) => (
                                <div key={i} className="space-y-2">
                                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                        {getIconForCategory(stack.category)}
                                        {stack.category}
                                    </h3>
                                    <div className="flex flex-wrap gap-2">
                                        {stack.skills.map((skill, j) => (
                                            <span key={j} className="px-3 py-1.5 bg-secondary text-secondary-foreground rounded-md text-sm font-medium border border-border/50">
                                                {skill}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                </div>

                {/* 4. Focus Areas */}
                <section className="space-y-6">
                    <h2 className="text-2xl font-bold flex items-center gap-2">
                        <Target className="text-primary" />
                        집중 공략 포인트 (Focus Areas)
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {job.focus_areas.map((area, i) => (
                            <div key={i} className="p-6 bg-gradient-to-br from-primary/5 to-transparent border border-primary/20 rounded-xl flex items-center justify-center text-center font-semibold text-foreground/90 shadow-sm">
                                {area}
                            </div>
                        ))}
                    </div>
                </section>

                {/* 5. Detailed Roadmap (Gamified) */}
                <section className="space-y-8 pt-8 border-t border-border">
                    <div className="text-center space-y-2 max-w-2xl mx-auto mb-8">
                        <h2 className="text-3xl font-bold flex items-center justify-center gap-3">
                            <MapIcon className="text-primary" size={32} />
                            학습 로드맵 (Roadmap)
                        </h2>
                        <p className="text-muted-foreground">
                            기초부터 실전 프로젝트까지, 퀴즈를 풀며 단계를 완성해보세요!
                        </p>
                    </div>

                    <JobReadinessTracker jobId={job.id} roadmap={job.roadmap} />
                </section>

                {/* 6. FAQ */}
                <section className="space-y-8 pt-8 border-t border-border max-w-3xl mx-auto">
                    <h2 className="text-2xl font-bold flex items-center justify-center gap-2">
                        <HelpCircle className="text-primary" />
                        자주 묻는 질문 (FAQ)
                    </h2>
                    <div className="grid gap-4">
                        {job.faq.map((item, i) => (
                            <div key={i} className="bg-card border border-border rounded-xl p-6 hover:bg-secondary/10 transition-colors">
                                <h3 className="font-bold text-lg mb-2 flex items-start gap-2">
                                    <span className="text-primary">Q.</span> {item.question}
                                </h3>
                                <p className="text-muted-foreground pl-6 leading-relaxed">
                                    {item.answer}
                                </p>
                            </div>
                        ))}
                    </div>
                </section>

                {/* 7. CTA */}
                <div className="flex justify-center pt-12 pb-20">
                    <Link href="/test">
                        <Button size="lg" className="px-12 py-6 text-xl rounded-full shadow-2xl shadow-primary/30 hover:shadow-primary/50 transition-all hover:scale-105 animate-pulse">
                            나에게 맞는 직무인지 테스트하기 →
                        </Button>
                    </Link>
                </div>

            </div>
        </div>
    );
}

// --- Helper Components ---

function StatCard({ icon, label, value, highlight = false }: { icon: React.ReactNode, label: string, value: string, highlight?: boolean }) {
    return (
        <div className={clsx("bg-card border rounded-xl p-6 flex flex-col items-center justify-center text-center gap-2 transition-all hover:border-primary/50", highlight ? "border-primary/50 bg-primary/5" : "border-border")}>
            <div className="p-3 bg-background rounded-full border border-border shadow-sm mb-1">
                {icon}
            </div>
            <div className="text-sm text-muted-foreground font-medium">{label}</div>
            <div className="text-lg font-bold text-foreground">{value}</div>
        </div>
    );
}

function getIconForCategory(category: string) {
    const map: Record<string, any> = {
        'Language': Code2,
        'Framework': Layers,
        'Database': Database,
        'AI Tools': Cpu,
        'Ops': Database,
    };
    const Icon = map[category] || CheckCircle2;
    return <Icon size={14} className="text-muted-foreground" />;
}
