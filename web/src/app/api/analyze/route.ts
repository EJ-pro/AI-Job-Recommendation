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
                    content: 'You are a warm and encouraging career counselor for a bootcamp student who has only 3 months left. Provide a concise (2-3 sentences) encouragement and specific advice on what to focus on for the recommended job. Speak in Korean.',
                },
                {
                    role: 'user',
                    content: `추천받은 직무: ${jobTitle}\n남은 3개월 동안 이 직무 취업을 위해 가장 집중해야 할 학습 포인트 1가지와 격려의 말을 해줘.`,
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
