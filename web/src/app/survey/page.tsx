'use client';

import React, { useState } from 'react';
import Button from '@/components/Button';
import { CheckCircle2, MessageSquare, Star, Send } from 'lucide-react';
import clsx from 'clsx';
import { useRouter } from 'next/navigation';

export default function SurveyPage() {
    const router = useRouter();
    const [submitted, setSubmitted] = useState(false);
    const [rating, setRating] = useState<number>(0);
    const [features, setFeatures] = useState<string[]>([]);
    const [feedback, setFeedback] = useState('');

    const toggleFeature = (feature: string) => {
        if (features.includes(feature)) {
            setFeatures(features.filter(f => f !== feature));
        } else {
            setFeatures([...features, feature]);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        // Here you would typically send data to your backend
        console.log({ rating, features, feedback });

        // Simulation
        setTimeout(() => {
            setSubmitted(true);
        }, 500);
    };

    if (submitted) {
        return (
            <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
                <div className="bg-white max-w-md w-full p-8 rounded-2xl shadow-xl text-center space-y-6">
                    <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                        <CheckCircle2 className="w-10 h-10 text-green-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-900">소중한 의견 감사합니다!</h2>
                    <p className="text-slate-600">
                        보내주신 피드백을 바탕으로<br />
                        더 나은 서비스를 만들겠습니다.
                    </p>
                    <Button onClick={() => router.push('/')} className="w-full bg-blue-600 hover:bg-blue-700">
                        메인으로 돌아가기
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-xl mx-auto">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-slate-900 flex items-center justify-center gap-3">
                        <MessageSquare className="w-8 h-8 text-blue-600" />
                        서비스 만족도 설문조사
                    </h1>
                    <p className="mt-2 text-slate-600">
                        AI Job Recommendation 서비스를 이용해주셔서 감사합니다.<br />
                        여러분의 솔직한 의견을 들려주세요.
                    </p>
                </div>

                <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 md:p-8 space-y-8">

                    {/* 1. Rating */}
                    <div className="space-y-3">
                        <label className="block text-sm font-semibold text-slate-900">
                            1. 전반적인 서비스 만족도는 어떠셨나요?
                        </label>
                        <div className="flex gap-2 justify-center py-4 bg-slate-50 rounded-xl">
                            {[1, 2, 3, 4, 5].map((star) => (
                                <button
                                    key={star}
                                    type="button"
                                    onClick={() => setRating(star)}
                                    className="p-1 transition-transform hover:scale-110 focus:outline-none"
                                >
                                    <Star
                                        className={clsx(
                                            "w-8 h-8 md:w-10 md:h-10 transition-colors",
                                            star <= rating ? "text-yellow-400 fill-yellow-400" : "text-slate-300"
                                        )}
                                    />
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* 2. Features */}
                    <div className="space-y-3">
                        <label className="block text-sm font-semibold text-slate-900">
                            2. 가장 유용했던 기능은 무엇인가요? (복수 선택 가능)
                        </label>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {[
                                '직무 적합도 테스트',
                                '직무별 상세 로드맵',
                                '직무별 연봉 및 상세 정보',
                                '전체 직무 탐색 (Jobs)',
                                '퀴즈 및 학습 트래킹'
                            ].map((feature) => (
                                <button
                                    key={feature}
                                    type="button"
                                    onClick={() => toggleFeature(feature)}
                                    className={clsx(
                                        "px-4 py-3 rounded-xl text-left text-sm font-medium transition-all border",
                                        features.includes(feature)
                                            ? "bg-blue-50 border-blue-500 text-blue-700 shadow-sm"
                                            : "bg-white border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50"
                                    )}
                                >
                                    {feature}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* 3. Feedback */}
                    <div className="space-y-3">
                        <label className="block text-sm font-semibold text-slate-900">
                            3. 더 바라는 점이나 개선할 부분이 있다면 알려주세요.
                        </label>
                        <textarea
                            value={feedback}
                            onChange={(e) => setFeedback(e.target.value)}
                            placeholder="자유롭게 작성해주세요..."
                            rows={4}
                            className="w-full px-4 py-3 rounded-xl bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white resize-none transition-all"
                        />
                    </div>

                    <Button
                        type="submit"
                        disabled={rating === 0}
                        className={clsx(
                            "w-full py-4 text-lg font-semibold shadow-lg shadow-blue-500/30 flex items-center justify-center gap-2",
                            "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 transform hover:-translate-y-1 transition-all"
                        )}
                    >
                        <Send className="w-5 h-5" />
                        의견 제출하기
                    </Button>
                </form>
            </div>
        </div>
    );
}
