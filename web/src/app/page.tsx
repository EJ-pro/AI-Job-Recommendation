import Button from '@/components/Button';
import JobRoleList from '@/components/JobRoleList';
import Link from 'next/link';
import { JOBS } from '@/lib/data';
import { ArrowRight, Sparkles, BrainCircuit, Target } from 'lucide-react'; // Need to install lucide-react if not present, or use text/emoji

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 md:p-12 space-y-16 max-w-5xl mx-auto">
      {/* Hero Section */}
      <section className="text-center space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary font-semibold text-sm mb-4 border border-primary/20">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
          </span>
          AI 커리어 가이드
        </div>

        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-foreground bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70">
          나에게 딱 맞는 <br className="md:hidden" />
          <span className="text-primary bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">AI 직무</span>는?
        </h1>

        <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
          AI 부트캠프 수강중 어떤 길로 가야 할지 고민이신가요? <br />
          당신의 성향과 강점을 분석해 <span className="text-foreground font-semibold">최적의 직무</span>를 추천해 드립니다.
        </p>

        <div className="pt-8">
          <Link href="/test">
            <Button size="lg" className="text-xl px-10 py-6 shadow-xl shadow-primary/30 hover:shadow-primary/50 transition-all hover:scale-105">
              테스트 시작하기
              {/* Simple arrow if Icon not available */}
              <span className="ml-2 text-2xl">→</span>
            </Button>
          </Link>
          <p className="mt-4 text-sm text-muted-foreground">
            ⏱️ 소요 시간: 약 5분 | 🔒 로그인 필요 없음
          </p>
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full animate-in fade-in slide-in-from-bottom-12 duration-1000 delay-200">
        {[
          { icon: '🎯', title: '직무 매칭', desc: '개발 성향과 관심사를 분석하여 핵심 AI 직무 중 최적의 포지션을 추천합니다.' },
          { icon: '⚡', title: '빠른 분석', desc: '고민할 필요 없이 5분 안에 결과를 확인하세요. 복잡한 절차 없이 바로 시작할 수 있습니다.' },
          { icon: '🚀', title: '취업 가이드', desc: '단순 추천을 넘어, 해당 직무를 위해 당장 무엇을 준비해야 할지 구체적인 가이드를 제공합니다.' },
        ].map((feature, i) => (
          <div key={i} className="p-8 rounded-2xl bg-card border border-border shadow-lg hover:shadow-xl transition-all hover:-translate-y-1">
            <div className="text-4xl mb-4">{feature.icon}</div>
            <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
            <p className="text-muted-foreground leading-relaxed">{feature.desc}</p>
          </div>
        ))}
      </section>

      {/* Job Roles Introduction */}
      <section className="w-full space-y-10 animate-in fade-in slide-in-from-bottom-16 duration-1000 delay-300">
        <div className="text-center space-y-4">
          <h2 className="text-3xl md:text-4xl font-bold">다루는 핵심 직무</h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            현재 AI 시장에서 가장 수요가 많은 7가지 핵심 직무를 분석합니다.
          </p>
        </div>

        <JobRoleList jobs={JOBS} />
      </section>

      {/* Floating Login/MyPage Button (Simple implementation) */}
      <div className="fixed top-6 right-6 z-50">
        <Link href="/login">
          <Button variant="outline" className="bg-background/80 backdrop-blur shadow-md hover:shadow-lg">
            로그인 / 마이페이지
          </Button>
        </Link>
      </div>
    </div>
  );
}
