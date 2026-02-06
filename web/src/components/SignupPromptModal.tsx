'use client';

import { X, MessageSquare, TrendingUp, FileText } from 'lucide-react';
import Button from './Button';
import Link from 'next/link';

interface SignupPromptModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SignupPromptModal({ isOpen, onClose }: SignupPromptModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="bg-background rounded-3xl max-w-lg w-full p-8 relative shadow-2xl border-2 border-primary/20 animate-in zoom-in-95 slide-in-from-bottom-4 duration-300">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
                >
                    <X className="w-6 h-6" />
                </button>

                {/* Content */}
                <div className="text-center space-y-6">
                    <div className="space-y-2">
                        <span className="inline-block px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider">
                            Exclusive Benefits
                        </span>
                        <h2 className="text-2xl md:text-3xl font-bold">
                            더 깊이 있는 분석이<br />
                            필요하신가요?
                        </h2>
                        <p className="text-muted-foreground">
                            무료 회원가입으로 전문가급 커리어 데이터를 확인하세요.
                        </p>
                    </div>

                    {/* Benefits List */}
                    <div className="space-y-4 text-left bg-muted/30 p-6 rounded-2xl">
                        <div className="flex items-start gap-4">
                            <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600 shrink-0">
                                <MessageSquare className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-bold">AI 커리어 컨설턴트 상담</h3>
                                <p className="text-sm text-muted-foreground">내 성향에 딱 맞는 1:1 맞춤 조언을 받아보세요.</p>
                            </div>
                        </div>

                        <div className="flex items-start gap-4">
                            <div className="w-10 h-10 rounded-xl bg-green-100 flex items-center justify-center text-green-600 shrink-0">
                                <TrendingUp className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-bold">직군별 연봉 & 수요 데이터</h3>
                                <p className="text-sm text-muted-foreground">실제 현업 데이터를 기반으로 한 연봉 정보를 공개합니다.</p>
                            </div>
                        </div>

                        <div className="flex items-start gap-4">
                            <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center text-purple-600 shrink-0">
                                <FileText className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-bold">상세 학습 로드맵 PDF</h3>
                                <p className="text-sm text-muted-foreground">지금 당장 시작할 수 있는 월별 실천 가이드를 드려요.</p>
                            </div>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="space-y-3 pt-2">
                        <Link href="/login" className="block w-full">
                            <Button fullWidth size="lg">
                                3초 만에 회원가입하고 전체 보기
                            </Button>
                        </Link>
                        <button
                            onClick={onClose}
                            className="text-xs text-muted-foreground hover:text-foreground underline-offset-4 hover:underline"
                        >
                            괜찮습니다. 현재 결과만 볼게요.
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
