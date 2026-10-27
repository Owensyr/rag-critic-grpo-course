"""Rebuild standalone figures from recorded scores and trainer telemetry."""
from common import ROOT,load
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    results=ROOT/'results';rows=load(results/'kaggle_scores.json')['experiments']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    labels=['Qwen baseline','+ Frozen CLM','+ SFT CLM','+ SFT CLM\n+ GRPO']
    values=[r['score'] for r in rows]
    bars=ax.bar(labels,values,color=['#9aa9bc','#7498bd','#32648d','#1e8378'],width=.6)
    ax.set_ylim(0,.60);ax.set_ylabel('Kaggle score');ax.set_title('2WikiMultiHopQA: full-rationale submissions')
    ax.bar_label(bars,labels=[f'{v:.5f}' for v in values],padding=5)
    ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    fig.text(.5,-.025,'Scores reported by experiment owner; hidden scoring split not independently verified.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(results/f'kaggle_scores.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    records=load(results/'sft_loss.json')['log_history']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for key,label,color in [('loss','Training loss','#32648d'),('eval_loss','Validation loss','#b76739')]:
        selected=[r for r in records if key in r]
        ax.plot([r['step'] for r in selected],[r[key] for r in selected],label=label,color=color,
                marker='o' if key=='eval_loss' else None,markersize=4,linewidth=1.6)
    ax.set_xlabel('Optimizer step');ax.set_ylabel('Cross-entropy loss');ax.set_title('CLM SFT: original 10K corpus')
    ax.grid(True,alpha=.18);ax.legend(frameon=False)
    for ext in ['png','svg']:fig.savefig(results/f'sft_loss.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig);print('FIGURES_READY')
if __name__=='__main__':main()
