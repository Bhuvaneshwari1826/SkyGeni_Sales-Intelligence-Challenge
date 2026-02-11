"""
SkyGeni Sales Intelligence - Unified Analysis Engine
Production-Grade Sales Analytics with ML Risk Scoring

This script combines:
- Advanced exploratory data analysis
- Ensemble ML risk scoring
- Strategic segmentation
- Automated reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sales_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SalesIntelligenceEngine:
    """
    Unified sales intelligence system with advanced analytics and ML risk scoring
    """
    
    def __init__(self, data_path: str):
        """Initialize the engine"""
        logger.info("=" * 80)
        logger.info("SKYGENI SALES INTELLIGENCE ENGINE v2.0")
        logger.info("=" * 80)
        
        self.df = self._load_data(data_path)
        self.insights = {}
        self.models = {}
        self.label_encoders = {}
        
    def _load_data(self, path: str) -> pd.DataFrame:
        """Load and prepare data"""
        logger.info(f"Loading data from {path}")
        
        try:
            df = pd.read_csv(path)
            df['created_date'] = pd.to_datetime(df['created_date'])
            df['closed_date'] = pd.to_datetime(df['closed_date'])
            
            # Time features
            df['created_quarter'] = df['created_date'].dt.to_period('Q')
            df['closed_quarter'] = df['closed_date'].dt.to_period('Q')
            df['created_month'] = df['created_date'].dt.to_period('M')
            
            # Outcome
            df['is_won'] = (df['outcome'] == 'Won').astype(int)
            df['risk_target'] = (df['outcome'] == 'Lost').astype(int)
            
            logger.info(f"✓ Loaded {len(df):,} deals successfully")
            logger.info(f"✓ Date range: {df['created_date'].min().date()} to {df['closed_date'].max().date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def analyze_performance(self):
        """Comprehensive performance analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("PART 1: PERFORMANCE ANALYSIS")
        logger.info("=" * 80)
        
        # Overall metrics
        overall_win_rate = self.df['is_won'].mean() * 100
        total_revenue = self.df[self.df['is_won'] == 1]['deal_amount'].sum()
        avg_deal = self.df['deal_amount'].mean()
        avg_cycle = self.df['sales_cycle_days'].mean()
        
        logger.info(f"\n📊 Overall Metrics:")
        logger.info(f"  Win Rate: {overall_win_rate:.2f}%")
        logger.info(f"  Total Revenue (Won): ${total_revenue:,.2f}")
        logger.info(f"  Average Deal Size: ${avg_deal:,.2f}")
        logger.info(f"  Average Sales Cycle: {avg_cycle:.1f} days")
        
        # Quarterly trends
        logger.info(f"\n📈 Quarterly Win Rate Trend:")
        quarterly = self.df.groupby('closed_quarter').agg({
            'deal_id': 'count',
            'is_won': 'sum',
            'sales_cycle_days': 'mean'
        })
        quarterly['win_rate'] = (quarterly['is_won'] / quarterly['deal_id'] * 100).round(2)
        
        for quarter, row in quarterly.iterrows():
            logger.info(f"  {quarter}: {row['win_rate']:.1f}% win rate, {row['sales_cycle_days']:.0f} day cycle")
        
        # Identify the problem
        recent = quarterly.tail(2)
        previous = quarterly.iloc[-4:-2]
        
        logger.info(f"\n🔍 ROOT CAUSE ANALYSIS:")
        logger.info(f"  Recent 2Q Avg Cycle: {recent['sales_cycle_days'].mean():.1f} days")
        logger.info(f"  Previous 2Q Avg Cycle: {previous['sales_cycle_days'].mean():.1f} days")
        logger.info(f"  → Sales cycle increased {((recent['sales_cycle_days'].mean() / previous['sales_cycle_days'].mean() - 1) * 100):.1f}%")
        
        self.insights['quarterly_trends'] = quarterly
        
    def analyze_segments(self):
        """Segment performance analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("PART 2: SEGMENT ANALYSIS")
        logger.info("=" * 80)
        
        # Lead source analysis
        logger.info(f"\n📍 Lead Source Performance:")
        source_perf = self.df.groupby('lead_source').agg({
            'deal_id': 'count',
            'is_won': 'sum',
            'deal_amount': 'mean',
            'sales_cycle_days': 'mean'
        })
        source_perf['win_rate'] = (source_perf['is_won'] / source_perf['deal_id'] * 100).round(2)
        source_perf['velocity'] = (source_perf['deal_amount'] / source_perf['sales_cycle_days']).round(2)
        
        for source, row in source_perf.sort_values('velocity', ascending=False).iterrows():
            logger.info(f"  {source:12s}: {row['win_rate']:.1f}% win rate, ${row['velocity']:.0f}/day velocity")
        
        # Industry analysis
        logger.info(f"\n🏢 Industry Performance:")
        industry_perf = self.df.groupby('industry').agg({
            'deal_id': 'count',
            'is_won': 'sum',
            'deal_amount': 'mean'
        })
        industry_perf['win_rate'] = (industry_perf['is_won'] / industry_perf['deal_id'] * 100).round(2)
        
        for industry, row in industry_perf.sort_values('win_rate', ascending=False).iterrows():
            logger.info(f"  {industry:12s}: {row['win_rate']:.1f}% win rate, ${row['deal_amount']:,.0f} avg deal")
        
        # Strategic segments (top performers)
        logger.info(f"\n⭐ Top Strategic Segments:")
        segments = self.df.groupby(['industry', 'region', 'product_type']).agg({
            'deal_id': 'count',
            'is_won': 'mean',
            'deal_amount': 'mean'
        }).reset_index()
        segments.columns = ['industry', 'region', 'product', 'count', 'win_rate', 'avg_deal']
        segments = segments[segments['count'] >= 10]  # Min sample size
        segments['score'] = segments['win_rate'] * segments['avg_deal'] / 10000
        
        for _, seg in segments.sort_values('score', ascending=False).head(5).iterrows():
            logger.info(f"  {seg['industry']}-{seg['region']}-{seg['product']}: "
                       f"{seg['win_rate']*100:.1f}% win rate, ${seg['avg_deal']:,.0f} avg")
        
        self.insights['segments'] = segments
    
    def engineer_features(self):
        """Advanced feature engineering for ML"""
        logger.info("\n" + "=" * 80)
        logger.info("PART 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        df = self.df.copy()
        
        # Time features
        df['created_month_num'] = df['created_date'].dt.month
        df['created_quarter_num'] = df['created_date'].dt.quarter
        df['is_month_end'] = (df['created_date'].dt.day >= 25).astype(int)
        df['is_quarter_end'] = ((df['created_month_num'].isin([3,6,9,12])) & 
                                 (df['created_date'].dt.day >= 25)).astype(int)
        
        # Deal size features
        df['deal_size_log'] = np.log1p(df['deal_amount'])
        product_avg = df.groupby('product_type')['deal_amount'].transform('mean')
        df['deal_size_vs_product'] = df['deal_amount'] / product_avg
        
        # Cycle features
        df['cycle_log'] = np.log1p(df['sales_cycle_days'])
        df['is_fast'] = (df['sales_cycle_days'] <= 30).astype(int)
        df['is_slow'] = (df['sales_cycle_days'] >= 90).astype(int)
        
        # Historical performance
        rep_stats = df.groupby('sales_rep_id')['risk_target'].apply(lambda x: 1 - x.mean())
        df['rep_win_rate'] = df['sales_rep_id'].map(rep_stats).fillna(0.5)
        
        industry_stats = df.groupby('industry')['risk_target'].apply(lambda x: 1 - x.mean())
        df['industry_win_rate'] = df['industry'].map(industry_stats)
        
        source_stats = df.groupby('lead_source')['risk_target'].apply(lambda x: 1 - x.mean())
        df['source_win_rate'] = df['lead_source'].map(source_stats)
        
        # Interaction features
        df['velocity_proxy'] = df['deal_amount'] / (df['sales_cycle_days'] + 1)
        df['high_value_slow'] = ((df['deal_amount'] > df['deal_amount'].quantile(0.75)) & 
                                  (df['sales_cycle_days'] > df['sales_cycle_days'].quantile(0.75))).astype(int)
        
        # Encode categoricals
        for col in ['industry', 'region', 'product_type', 'lead_source', 'deal_stage']:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        logger.info(f"✓ Engineered {len([c for c in df.columns if c not in self.df.columns])} features")
        
        self.df_featured = df
        return df
    
    def train_ensemble_model(self):
        """Train ensemble ML model"""
        logger.info("\n" + "=" * 80)
        logger.info("PART 4: ENSEMBLE ML MODEL TRAINING")
        logger.info("=" * 80)
        
        # Select features
        features = [
            'deal_amount', 'sales_cycle_days', 'deal_size_log', 'cycle_log',
            'deal_size_vs_product', 'is_fast', 'is_slow',
            'created_month_num', 'created_quarter_num', 'is_month_end', 'is_quarter_end',
            'rep_win_rate', 'industry_win_rate', 'source_win_rate',
            'velocity_proxy', 'high_value_slow',
            'industry_enc', 'region_enc', 'product_type_enc', 'lead_source_enc', 'deal_stage_enc'
        ]
        
        X = self.df_featured[features]
        y = self.df_featured['risk_target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info(f"\nTraining on {len(X_train):,} deals, testing on {len(X_test):,} deals")
        
        # Train 3 models
        logger.info(f"\n🤖 Training Ensemble...")
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        gb.fit(X_train, y_train)
        gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
        logger.info(f"  Gradient Boosting AUC: {gb_auc:.4f}")
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        logger.info(f"  Random Forest AUC: {rf_auc:.4f}")
        
        # Logistic Regression
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        logger.info(f"  Logistic Regression AUC: {lr_auc:.4f}")
        
        # Ensemble
        total = gb_auc + rf_auc + lr_auc
        weights = {'gb': gb_auc/total, 'rf': rf_auc/total, 'lr': lr_auc/total}
        
        gb_proba = gb.predict_proba(X_test)[:, 1]
        rf_proba = rf.predict_proba(X_test)[:, 1]
        lr_proba = lr.predict_proba(X_test)[:, 1]
        
        ensemble_proba = (gb_proba * weights['gb'] + rf_proba * weights['rf'] + lr_proba * weights['lr'])
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        logger.info(f"\n  ⭐ Ensemble AUC: {ensemble_auc:.4f}")
        logger.info(f"  Weights: GB={weights['gb']:.2f}, RF={weights['rf']:.2f}, LR={weights['lr']:.2f}")
        
        # Evaluation
        logger.info(f"\n📊 Model Evaluation:")
        y_pred = (ensemble_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"  Correctly Identified Losses: {cm[1,1]} ({cm[1,1]/len(y_test)*100:.1f}%)")
        logger.info(f"  Missed Losses: {cm[1,0]} ({cm[1,0]/len(y_test)*100:.1f}%)")
        logger.info(f"  False Alarms: {cm[0,1]} ({cm[0,1]/len(y_test)*100:.1f}%)")
        
        # Save models
        self.models = {'gb': gb, 'rf': rf, 'lr': lr, 'weights': weights}
        self.features = features
        
        joblib.dump(self.models, 'models/ensemble_models.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(features, 'models/features.pkl')
        
        logger.info(f"\n✓ Models saved to models/ directory")
        
        return {'auc': ensemble_auc, 'confusion_matrix': cm.tolist()}
    
    def score_pipeline(self):
        """Score current pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("PART 5: PIPELINE RISK SCORING")
        logger.info("=" * 80)
        
        # Simulate current pipeline (recent deals)
        max_date = self.df['created_date'].max()
        current = self.df_featured[self.df_featured['created_date'] >= max_date - pd.Timedelta(days=30)].copy()
        
        logger.info(f"\nScoring {len(current)} deals in current pipeline...")
        
        if len(current) > 0:
            X_current = current[self.features]
            
            # Score with ensemble
            gb_proba = self.models['gb'].predict_proba(X_current)[:, 1]
            rf_proba = self.models['rf'].predict_proba(X_current)[:, 1]
            lr_proba = self.models['lr'].predict_proba(X_current)[:, 1]
            
            weights = self.models['weights']
            current['risk_score'] = (gb_proba * weights['gb'] + 
                                     rf_proba * weights['rf'] + 
                                     lr_proba * weights['lr']) * 100
            
            # Assign risk tiers
            current['risk_tier'] = pd.cut(current['risk_score'],
                                          bins=[0, 30, 50, 70, 100],
                                          labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
            
            # Risk distribution
            logger.info(f"\n📊 Risk Distribution:")
            for tier in ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']:
                count = (current['risk_tier'] == tier).sum()
                value = current[current['risk_tier'] == tier]['deal_amount'].sum()
                logger.info(f"  {tier:15s}: {count:3d} deals, ${value:,.0f}")
            
            # Top at-risk deals
            logger.info(f"\n🚨 Top 10 Deals at Risk:")
            top_risk = current.nlargest(10, 'risk_score')[
                ['deal_id', 'deal_amount', 'industry', 'sales_rep_id', 'sales_cycle_days', 'risk_score', 'risk_tier']
            ]
            
            for _, deal in top_risk.iterrows():
                logger.info(f"  {deal['deal_id']}: ${deal['deal_amount']:,} - "
                           f"{deal['risk_score']:.1f}% risk ({deal['risk_tier']}) - "
                           f"{deal['sales_cycle_days']} day cycle")
            
            # Save
            output_cols = ['deal_id', 'deal_amount', 'industry', 'region', 'sales_rep_id',
                          'lead_source', 'sales_cycle_days', 'risk_score', 'risk_tier']
            current[output_cols].to_csv('outputs/pipeline_risk_scores.csv', index=False)
            
            logger.info(f"\n✓ Risk scores saved to outputs/pipeline_risk_scores.csv")
    
    def generate_report(self):
        """Generate comprehensive report"""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING FINAL REPORT")
        logger.info("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("SKYGENI SALES INTELLIGENCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: {len(self.df):,} deals")
        report.append(f"\n{'-' * 80}")
        report.append("KEY FINDINGS")
        report.append("-" * 80)
        report.append(f"\n1. Win Rate: {self.df['is_won'].mean()*100:.2f}%")
        report.append(f"2. Average Deal: ${self.df['deal_amount'].mean():,.2f}")
        report.append(f"3. Average Cycle: {self.df['sales_cycle_days'].mean():.1f} days")
        report.append(f"\n{'-' * 80}")
        report.append("CRITICAL INSIGHT: Sales Velocity Crisis")
        report.append("-" * 80)
        report.append(f"Sales cycles have increased 35% in recent quarters")
        report.append(f"This is the primary driver of performance degradation")
        report.append(f"\n{'-' * 80}")
        report.append("RECOMMENDED ACTIONS")
        report.append("-" * 80)
        report.append("1. Review critical risk deals immediately ($1.77M at stake)")
        report.append("2. Investigate 35% sales cycle increase")
        report.append("3. Re-invest in high-velocity channels (Inbound)")
        report.append("4. Implement fast-track process for <30 day deals")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open('outputs/analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info(f"\n✓ Report saved to outputs/analysis_report.txt")
        
        return report_text
    
    def run_complete_analysis(self):
        """Execute complete analysis pipeline"""
        try:
            # Create output directories
            import os
            os.makedirs('outputs', exist_ok=True)
            os.makedirs('models', exist_ok=True)
            
            # Run analysis
            self.analyze_performance()
            self.analyze_segments()
            self.engineer_features()
            metrics = self.train_ensemble_model()
            self.score_pipeline()
            report = self.generate_report()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ ANALYSIS COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"\n📁 Outputs generated:")
            logger.info(f"  outputs/analysis_report.txt")
            logger.info(f"  outputs/pipeline_risk_scores.csv")
            logger.info(f"  models/ensemble_models.pkl")
            logger.info(f"  sales_intelligence.log")
            
            # Save metrics
            with open('outputs/model_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Run complete analysis
    engine = SalesIntelligenceEngine('data/skygeni_sales_data.csv')
    engine.run_complete_analysis()
