import { NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
    try {
        const { messages } = await req.json();

        if (!messages || !Array.isArray(messages)) {
            return NextResponse.json(
                { error: 'Messages array is required.' },
                { status: 400 }
            );
        }

        const systemPrompt = `
    You are an expert AI Career Consultant for bootcamp students and junior developers.
    Your goal is to provide personalized, actionable career advice and study strategies.
    
    Guidelines:
    - Listen carefully to the user's situation.
    - Provide specific, step-by-step guidance.
    - Be encouraging but realistic.
    - If the user's input is vague, ask clarifying questions to give better advice.
    - Keep responses concise and easy to read (use bullet points).
    - Tone: Professional, warm, supportive, like a senior mentor.
    - Language: Korean.
    `;

        const completion = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: [
                { role: 'system', content: systemPrompt },
                ...messages
            ],
            max_tokens: 1000,
        });

        const reply = completion.choices[0].message.content;

        return NextResponse.json({ reply });
    } catch (error) {
        console.error('OpenAI API Error:', error);
        return NextResponse.json(
            { error: 'Failed to generate advice. Please try again later.' },
            { status: 500 }
        );
    }
}
