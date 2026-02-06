'use client';

import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import Button from '@/components/Button';
import { User, Mail, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export default function LoginPage() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const { login } = useAuth();

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!name.trim() || !email.trim()) return;
        login(name, email);
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-b from-background to-secondary/20">
            <div className="w-full max-w-md bg-card p-8 rounded-3xl border border-border shadow-xl space-y-8">
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold">환영합니다! 👋</h1>
                    <p className="text-muted-foreground">
                        AI 커리어 챗봇을 이용하려면 로그인이 필요합니다.
                    </p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-sm font-medium ml-1">이름</label>
                        <div className="relative">
                            <User className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground h-5 w-5" />
                            <input
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="홍길동"
                                className="w-full pl-12 pr-4 py-3 rounded-xl border border-input bg-background focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all outline-none"
                                required
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium ml-1">이메일</label>
                        <div className="relative">
                            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground h-5 w-5" />
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="hello@example.com"
                                className="w-full pl-12 pr-4 py-3 rounded-xl border border-input bg-background focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all outline-none"
                                required
                            />
                        </div>
                    </div>

                    <Button type="submit" className="w-full text-lg py-6 shadow-lg hover:shadow-primary/20">
                        시작하기
                        <ArrowRight className="ml-2 h-5 w-5" />
                    </Button>
                </form>

                <div className="text-center">
                    <Link href="/" className="text-sm text-muted-foreground hover:text-primary transition-colors underline-offset-4 hover:underline">
                        메인으로 돌아가기
                    </Link>
                </div>
            </div>
        </div>
    );
}
