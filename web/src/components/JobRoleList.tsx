'use client';

import { useState } from 'react';
import { JobRole } from '@/lib/data';
import {
    BrainCircuit,
    Code,
    Database,
    BarChart3,
    Search,
    Presentation,
    Settings,
    ChevronDown,
    ChevronUp,
    Map
} from 'lucide-react';

interface JobRoleListProps {
    jobs: JobRole[];
}

const iconMap: Record<string, any> = {
    'ai-app': Code,
    'prompt-eng': Search,
    'mlops': Settings,
    'data-eng': Database,
    'data-sci': BarChart3,
    'research': BrainCircuit,
    'pm': Presentation,
    'ml-eng': BrainCircuit, // Reusing BrainCircuit or finding a better one
};

export default function JobRoleList({ jobs }: JobRoleListProps) {
    const [expandedId, setExpandedId] = useState<string | null>(null);

    const toggleJob = (id: string) => {
        setExpandedId(expandedId === id ? null : id);
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {jobs.map((job) => {
                const Icon = iconMap[job.id] || BrainCircuit;
                const isExpanded = expandedId === job.id;

                return (
                    <div
                        key={job.id}
                        className={`group relative p-6 rounded-2xl bg-card border transition-all duration-300 cursor-pointer overflow-hidden
                            ${isExpanded
                                ? 'border-primary shadow-lg shadow-primary/10 ring-1 ring-primary row-span-2'
                                : 'border-border/50 hover:border-primary/50 hover:shadow-md hover:-translate-y-1'
                            }`}
                        onClick={() => toggleJob(job.id)}
                    >
                        <div className="relative space-y-4">
                            <div className="flex items-start justify-between">
                                <div className={`p-3 rounded-xl transition-colors ${isExpanded ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground group-hover:bg-primary/10 group-hover:text-primary'}`}>
                                    <Icon size={24} />
                                </div>
                                <div className={`transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`}>
                                    <ChevronDown size={20} className="text-muted-foreground" />
                                </div>
                            </div>

                            <div>
                                <h3 className={`text-xl font-bold mb-2 transition-colors ${isExpanded ? 'text-primary' : 'text-foreground'}`}>
                                    {job.title}
                                </h3>
                                <p className="text-muted-foreground text-sm leading-relaxed">
                                    {job.description}
                                </p>
                            </div>

                            {!isExpanded && (
                                <div className="flex flex-wrap gap-2 pt-2">
                                    {job.tags.map((tag) => (
                                        <span key={tag} className="text-xs px-2.5 py-1 rounded-full bg-secondary/50 text-secondary-foreground border border-secondary/20">
                                            #{tag}
                                        </span>
                                    ))}
                                </div>
                            )}

                            {/* Expanded Content */}
                            <div className={`grid grid-rows-[0fr] transition-[grid-template-rows] duration-300 ${isExpanded ? 'grid-rows-[1fr]' : ''}`}>
                                <div className="overflow-hidden">
                                    <div className="pt-6 space-y-6 border-t border-border/50 mt-4">
                                        {/* Focus Areas */}
                                        <div className="space-y-3">
                                            <h4 className="font-semibold flex items-center gap-2 text-sm text-foreground/80">
                                                <Target size={16} /> 3개월 집중 공략
                                            </h4>
                                            <ul className="space-y-2">
                                                {job.focus_areas.map((area, idx) => (
                                                    <li key={idx} className="bg-secondary/30 px-3 py-2 rounded-lg text-sm flex items-center gap-2">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                                                        {area}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Tasks */}
                                        <div className="space-y-3">
                                            <h4 className="font-semibold flex items-center gap-2 text-sm text-foreground/80">
                                                <Code size={16} /> 주요 업무
                                            </h4>
                                            <div className="flex flex-wrap gap-2">
                                                {job.tasks.map((task, idx) => (
                                                    <span key={idx} className="text-xs px-2 py-1 rounded-md border border-border bg-background">
                                                        {task}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Roadmap Preview */}
                                        <div className="space-y-3">
                                            <h4 className="font-semibold flex items-center gap-2 text-sm text-foreground/80">
                                                <Map size={16} /> 로드맵 미리보기
                                            </h4>
                                            <div className="space-y-3 pl-4 border-l-2 border-border relative">
                                                {job.roadmap.map((step, idx) => (
                                                    <div key={idx} className="relative">
                                                        <span className="absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full bg-muted-foreground/30 ring-4 ring-background" />
                                                        <p className="text-xs font-medium text-muted-foreground mb-0.5">{step.step}</p>
                                                        <p className="text-sm line-clamp-1 text-foreground/90">{step.action}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

function Target({ size, className }: { size?: number, className?: string }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" width={size || 24} height={size || 24} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="6" />
            <circle cx="12" cy="12" r="2" />
        </svg>
    );
} 
