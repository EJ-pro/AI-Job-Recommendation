'use client';

import React from 'react';
import { MessageSquareText } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function SurveyButton() {
    const router = useRouter();

    return (
        <button
            onClick={() => router.push('/survey')}
            className="fixed bottom-6 left-6 z-50 flex items-center justify-center p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg transition-all duration-300 hover:scale-110 group"
            aria-label="Take Survey"
        >
            <MessageSquareText className="w-6 h-6" />
            <span className="absolute left-full ml-3 px-3 py-1 bg-gray-900 text-white text-sm rounded-lg opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300 whitespace-nowrap pointer-events-none">
                의견 보내기
            </span>
        </button>
    );
}
