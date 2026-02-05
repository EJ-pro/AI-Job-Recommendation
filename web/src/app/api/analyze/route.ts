import { NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: Request) {
    try {
        const { jobTitle } = await request.json();

        if (!jobTitle) {
            return NextResponse.json({ error: 'Job title is required' }, { status: 400 });
        }

        const completion = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: [
                {
                    role: 'system',
                    content: 'You are a warm and encouraging career counselor for AI bootcamp graduates. Provide a concise (2-3 sentences) and personalized encouragement based on the recommended job title. Speak in Korean.',
                },
                {
                    role: 'user',
                    content: `추천받은 직무: ${jobTitle}\n이 직무를 추천받은 신입 지원자에게 해줄 수 있는 핵심 역량 조언과 격려의 말을 짧게 해줘.`,
                },
            ],
            max_tokens: 300,
        });

        const advice = completion.choices[0].message.content;

        return NextResponse.json({ result: advice });
    } catch (error: any) {
        console.error('OpenAI Error:', error);
        return NextResponse.json(
            { error: error?.message || 'Failed to generate advice' },
            { status: 500 }
        );
    }
}
