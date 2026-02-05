import { JobRole } from '../lib/data';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

interface JobCardProps {
    job: JobRole;
    rank: 1 | 2;
    className?: string;
}

export default function JobCard({ job, rank, className }: JobCardProps) {
    const isTop = rank === 1;

    return (
        <div
            className={twMerge(
                clsx(
                    'relative overflow-hidden rounded-2xl p-8 transition-all duration-500 hover:shadow-2xl',
                    isTop
                        ? 'bg-gradient-to-br from-primary/10 to-secondary/10 border-2 border-primary/20 shadow-xl shadow-primary/10'
                        : 'bg-card border border-border shadow-lg opacity-90 grayscale-[0.2] hover:grayscale-0 hover:opacity-100',
                    className
                )
            )}
        >
            {/* Rank Badge */}
            <div className={clsx(
                'absolute top-0 right-0 px-4 py-2 rounded-bl-2xl font-bold text-sm tracking-wide',
                isTop ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            )}>
                {isTop ? '✨ Best Match' : 'Runner-up'}
            </div>

            <div className="space-y-6 mt-2">
                <div>
                    <h3 className={clsx('font-bold tracking-tight', isTop ? 'text-3xl text-primary' : 'text-2xl text-foreground')}>
                        {job.title}
                    </h3>
                    <p className="mt-4 text-lg text-muted-foreground leading-relaxed">
                        {job.description}
                    </p>
                </div>

                <div className="space-y-3">
                    <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Key Tasks</h4>
                    <ul className="grid gap-2">
                        {job.tasks.map((task, i) => (
                            <li key={i} className="flex items-start gap-2 text-foreground/90">
                                <span className={isTop ? 'text-primary' : 'text-secondary'}>•</span>
                                {task}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="pt-4 flex flex-wrap gap-2">
                    {job.tags.map((tag) => (
                        <span
                            key={tag}
                            className={clsx(
                                'px-3 py-1 rounded-full text-sm font-medium border',
                                isTop
                                    ? 'bg-primary/5 text-primary border-primary/20'
                                    : 'bg-muted text-muted-foreground border-border'
                            )}
                        >
                            #{tag}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}
