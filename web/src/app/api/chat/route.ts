import { NextResponse } from 'next/server';
import OpenAI from 'openai';
import { JOBS } from '@/lib/data';

// OpenAI client initialization
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
    try {
        const { messages } = await req.json();

        // 1. Prepare Context from Data
        const jobContext = JOBS.map(job => `
[Job ID: ${job.id}]
- Title: ${job.title}
- Description: ${job.description}
- Salary: ${job.salary_range}
- Difficulty: ${job.difficulty}
- Responsibilities: ${job.responsibilities.join(', ')}
- Tech Stack: ${job.tech_stack.map(t => t.category + ': ' + t.skills.join(', ')).join(' | ')}
- Roadmap Phases: ${job.roadmap.map((r, i) => `${i + 1}. ${r.step} - ${r.title}`).join(', ')}
        `).join('\n\n');

        const systemPrompt = `
You are the AI Career Guide for the "AI Job Finder" website.
Your goal is to help users find their suitable AI career path and explain the job roles available on this site.
You have access to the following specific job data from our database:

${jobContext}

Instructions:
1. ONLY answer based on the provided context when asked about specific jobs, salaries, or roadmaps defined here.
2. If the user asks about a job NOT in our list, explain that this site currently focuses on the jobs listed above, but you can provide general advice.
3. Keep answers concise, friendly, and encouraging.
4. Use Korean language (한국어) primarily.
5. If asked about the roadmap, summarize the phases clearly.
        `;

        // 2. Call OpenAI API
        const completion = await openai.chat.completions.create({
            model: 'gpt-4o', // or gpt-3.5-turbo
            messages: [
                { role: 'system', content: systemPrompt },
                ...messages
            ],
            temperature: 0.7,
            max_tokens: 500,
        });

        const reply = completion.choices[0].message.content;

        return NextResponse.json({ reply });

    } catch (error: any) {
        console.error('OpenAI API Error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch response', details: error.message },
            { status: 500 }
        );
    }
}
