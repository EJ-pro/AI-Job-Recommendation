'use client';

import { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import AIConsultant from '@/components/AIConsultant';
import Button from '@/components/Button';
import { User, LogOut } from 'lucide-react';
import Link from 'next/link';
import StatsDashboard from '@/components/StatsDashboard';
import PersonalResultSummary from '@/components/PersonalResultSummary';

export default function MyPage() {
    const { user, isLoading, logout } = useAuth();
    const router = useRouter();

    useEffect(() => {
        if (!isLoading && !user) {
            router.push('/login');
        }
    }, [user, isLoading, router]);

    if (isLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background p-6 md:p-12">
            <div className="max-w-5xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-6 bg-card rounded-3xl border border-border shadow-sm">
                    <div className="flex items-center gap-4">
                        <div className="bg-primary/10 p-4 rounded-full">
                            <User className="h-8 w-8 text-primary" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold">{user.name}님, 안녕하세요!</h1>
                            <p className="text-muted-foreground">{user.email}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <Link href="/">
                            <Button variant="outline" className="border-border">메인으로</Button>
                        </Link>
                        <Button onClick={logout} variant="outline" className="text-destructive hover:bg-destructive/10 border-destructive/20">
                            <LogOut className="mr-2 h-4 w-4" />
                            로그아웃
                        </Button>
                    </div>
                </div>

                {/* AI Consultant Session */}
                <div className="space-y-4">
                    <h2 className="text-xl font-bold px-2 border-l-4 border-primary pl-4">나만의 AI 커리어 멘토</h2>
                    <AIConsultant />
                </div>

                {/* Statistics Dashboard */}
                <div className="space-y-4 pt-4">
                    <h2 className="text-xl font-bold px-2 border-l-4 border-primary pl-4">전체 참여자 데이터 분석</h2>
                    <StatsDashboard />
                </div>

                {/* Personal Result Summary */}
                <div className="space-y-4 pt-4">
                    <h2 className="text-xl font-bold px-2 border-l-4 border-primary pl-4">테스트 결과 종합</h2>
                    <PersonalResultSummary userEmail={user.email} />
                </div>
            </div>
        </div>
    );
}
