import { JOBS, JobRole, QUESTIONS } from './data';

export function calculateRecommendation(answers: Record<number, number>): JobRole[] {
    // answers: key is question ID, value is selected option index

    const scores: Record<string, number> = {};

    // Initialize scores
    JOBS.forEach((job) => {
        scores[job.id] = 0;
    });

    // Calculate scores
    Object.entries(answers).forEach(([questionId, optionIndex]) => {
        const qId = parseInt(questionId);
        const question = QUESTIONS.find((q) => q.id === qId);

        if (question && question.options[optionIndex]) {
            const weights = question.options[optionIndex].weights;
            Object.entries(weights).forEach(([jobId, weight]) => {
                if (scores[jobId] !== undefined && weight) {
                    scores[jobId] += weight;
                }
            });
        }
    });

    // Sort jobs by score (descending)
    const sortedJobs = JOBS.map((job) => ({
        ...job,
        score: scores[job.id] || 0,
    })).sort((a, b) => b.score - a.score);

    // Return top 2 matching jobs
    return sortedJobs.slice(0, 2);
}
