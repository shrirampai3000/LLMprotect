"""
FastAPI REST API Server for Cryptographic Intent Binding.

Endpoints:
- POST /detect - Analyze prompt for adversarial patterns
- POST /authorize - Generate authorization token
- POST /verify - Verify authorization token
- GET /audit - Get audit log summary
- POST /credentials - Create scoped credential
- GET /health - Health check
- Static files served at /static for demo UI
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
from pathlib import Path
import uvicorn

from .schemas import (
    DetectionRequest, DetectionResponse,
    BatchDetectionRequest, BatchDetectionResponse,
    AuthorizationRequest, AuthorizationResponse,
    VerificationRequest, VerificationResponse,
    AuditLogResponse, AuditEntryResponse,
    CredentialRequest, CredentialResponse,
    HealthResponse
)
from ..core.pipeline import IntentBindingPipeline
from ..core.credentials import CredentialManager
from ..crypto.signing import AuthorizationToken
from .. import __version__


# Global instances (initialized on startup)
_pipeline: Optional[IntentBindingPipeline] = None
_credential_manager: Optional[CredentialManager] = None


def get_pipeline() -> IntentBindingPipeline:
    """Dependency to get the pipeline instance."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def get_credential_manager() -> CredentialManager:
    """Dependency to get the credential manager."""
    if _credential_manager is None:
        raise HTTPException(status_code=503, detail="Credential manager not initialized")
    return _credential_manager


def create_app(
    ml_pipeline=None,
    key_storage_path: Optional[Path] = None,
    audit_log_path: Optional[Path] = None
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        ml_pipeline: Optional ML inference pipeline
        key_storage_path: Path for key storage
        audit_log_path: Path for audit logs
        
    Returns:
        Configured FastAPI app
    """
    global _pipeline, _credential_manager
    
    app = FastAPI(
        title="Cryptographic Intent Binding API",
        description="Adversarial Manipulation Detection for Agentic AI Systems",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Mount static files for demo UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Redirect root to demo page
    from fastapi.responses import RedirectResponse
    
    @app.get("/", include_in_schema=False)
    async def redirect_to_demo():
        """Redirect root to the demo UI."""
        return RedirectResponse(url="/static/index.html")
    
    @app.on_event("startup")
    async def startup():
        global _pipeline, _credential_manager
        
        from ..crypto.keys import KeyManager
        
        key_path = key_storage_path or Path("keys")
        audit_path = audit_log_path or Path("audit_logs")
        
        key_manager = KeyManager(key_path)
        
        _pipeline = IntentBindingPipeline(
            ml_pipeline=ml_pipeline,
            key_manager=key_manager,
            audit_log_path=audit_path
        )
        
        _credential_manager = CredentialManager()
        
        print(f"Intent Binding API started (version {__version__})")
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        pipeline = get_pipeline()
        
        return HealthResponse(
            status="healthy",
            version=__version__,
            ml_model_loaded=pipeline.ml_pipeline is not None,
            crypto_initialized=pipeline.key_manager.current_keypair is not None,
            audit_log_entries=len(pipeline.audit_log.entries)
        )
    
    @app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
    async def detect_adversarial(
        request: DetectionRequest,
        pipeline: IntentBindingPipeline = Depends(get_pipeline)
    ):
        """
        Analyze a prompt for adversarial patterns.
        
        Returns detection result with risk score and decision.
        """
        result = pipeline.process(
            prompt=request.prompt,
            action=request.action,
            target=request.target
        )
        
        return DetectionResponse(
            prompt_preview=result.prompt[:100] + "..." if len(result.prompt) > 100 else result.prompt,
            is_adversarial=result.is_adversarial,
            risk_score=result.risk_score,
            decision=result.decision,
            explanation=result.explanation,
            processing_time_ms=result.processing_time_ms,
            has_authorization=result.authorization_token is not None
        )
    
    @app.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
    async def detect_adversarial_batch(
        request: BatchDetectionRequest,
        pipeline: IntentBindingPipeline = Depends(get_pipeline)
    ):
        """Analyze multiple prompts for adversarial patterns."""
        import time
        start = time.time()
        
        results = []
        adversarial_count = 0
        
        for prompt in request.prompts:
            result = pipeline.process(
                prompt=prompt,
                action=request.action,
                target=request.target
            )
            
            if result.is_adversarial:
                adversarial_count += 1
            
            results.append(DetectionResponse(
                prompt_preview=result.prompt[:100] + "..." if len(result.prompt) > 100 else result.prompt,
                is_adversarial=result.is_adversarial,
                risk_score=result.risk_score,
                decision=result.decision,
                explanation=result.explanation,
                processing_time_ms=result.processing_time_ms,
                has_authorization=result.authorization_token is not None
            ))
        
        total_time = (time.time() - start) * 1000
        
        return BatchDetectionResponse(
            results=results,
            total_prompts=len(request.prompts),
            adversarial_count=adversarial_count,
            total_processing_time_ms=total_time
        )
    
    @app.post("/authorize", response_model=AuthorizationResponse, tags=["Authorization"])
    async def create_authorization(
        request: AuthorizationRequest,
        pipeline: IntentBindingPipeline = Depends(get_pipeline)
    ):
        """
        Generate a signed authorization token for a prompt.
        
        Use this after a prompt has been reviewed and approved.
        """
        token = pipeline.auth_manager.create_authorization(
            prompt=request.prompt,
            action=request.action,
            target=request.target
        )
        
        return AuthorizationResponse(
            prompt_hash=token.prompt_hash,
            action=token.action,
            target=token.target,
            timestamp=token.timestamp,
            expires_at=token.expires_at,
            nonce=token.nonce,
            signature=token.signature
        )
    
    @app.post("/verify", response_model=VerificationResponse, tags=["Authorization"])
    async def verify_authorization(
        request: VerificationRequest,
        pipeline: IntentBindingPipeline = Depends(get_pipeline)
    ):
        """Verify an authorization token."""
        token = AuthorizationToken(
            prompt_hash=request.prompt_hash,
            action=request.action,
            target=request.target,
            timestamp=request.timestamp,
            expires_at=request.expires_at,
            nonce=request.nonce,
            signature=request.signature
        )
        
        result = pipeline.verify_authorization(token, request.expected_prompt)
        
        return VerificationResponse(
            valid=result.get('valid', False),
            error=result.get('error'),
            expires_in=result.get('expires_in')
        )
    
    @app.get("/audit", response_model=AuditLogResponse, tags=["Audit"])
    async def get_audit_log(
        pipeline: IntentBindingPipeline = Depends(get_pipeline)
    ):
        """Get audit log summary and integrity status."""
        summary = pipeline.get_audit_summary()
        stats = summary.get('statistics', {})
        integrity = summary.get('integrity', {})
        
        return AuditLogResponse(
            total_entries=stats.get('total_entries', 0),
            decisions=stats.get('decisions', {}),
            avg_risk_score=stats.get('avg_risk_score', 0.0),
            merkle_root=stats.get('merkle_root'),
            integrity_valid=integrity.get('valid', True)
        )
    
    @app.post("/credentials", response_model=CredentialResponse, tags=["Credentials"])
    async def create_credential(
        request: CredentialRequest,
        cred_manager: CredentialManager = Depends(get_credential_manager)
    ):
        """Create a new scoped credential."""
        credential = cred_manager.create_credential(
            action_types=request.action_types,
            resources=request.resources,
            validity_minutes=request.validity_minutes,
            max_uses=request.max_uses
        )
        
        return CredentialResponse(
            credential_id=credential.credential_id,
            token=credential.token,
            action_types=list(credential.action_types),
            resources=list(credential.resources),
            expires_at=credential.expires_at.isoformat(),
            max_uses=credential.max_uses
        )
    
    @app.get("/evaluation", tags=["Evaluation"])
    async def get_evaluation_metrics():
        """
        Get model evaluation metrics for research paper.
        
        Returns comprehensive metrics including detection rates, F1 scores,
        confusion matrix, and latency measurements.
        """
        import json
        
        # Load both evaluation files
        results = {}
        
        # Training evaluation results
        eval_path = Path("checkpoints/evaluation_results.json")
        if eval_path.exists():
            with open(eval_path) as f:
                results["training_evaluation"] = json.load(f)
        
        # Paper-safe conservative metrics
        paper_path = Path("checkpoints/evaluation_results_paper.json")
        if paper_path.exists():
            with open(paper_path) as f:
                results["paper_metrics"] = json.load(f)
        
        # Model info
        model_path = Path("checkpoints/best_model.pt")
        if model_path.exists():
            import os
            results["model_info"] = {
                "model_file": "best_model.pt",
                "model_size_bytes": os.path.getsize(model_path),
                "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2)
            }
        
        # Add summary for easy access
        if "training_evaluation" in results:
            te = results["training_evaluation"]
            results["summary"] = {
                "detection_rate": f"{te.get('detection_rate', 0) * 100:.2f}%",
                "false_negative_rate": f"{te.get('false_negative_rate', 0) * 100:.2f}%",
                "false_positive_rate": f"{te.get('false_positive_rate', 0) * 100:.2f}%",
                "f1_score": f"{te.get('f1_score', 0):.4f}",
                "precision": f"{te.get('precision', 0):.4f}",
                "auc_roc": f"{te.get('auc_roc', 0):.4f}",
                "mean_latency_ms": f"{te.get('mean_latency_ms', 0):.2f}ms",
                "model_size_mb": f"{te.get('model_size_mb', 0):.2f}MB"
            }
        
        return results
    
    @app.get("/evaluation/html", include_in_schema=False)
    async def evaluation_page():
        """Serve the evaluation metrics HTML page."""
        from fastapi.responses import FileResponse
        eval_page = Path(__file__).parent / "static" / "evaluation.html"
        if eval_page.exists():
            return FileResponse(eval_page)
        else:
            from fastapi.responses import HTMLResponse
            return HTMLResponse("<h1>Evaluation page not found</h1>")
    
    return app


# Default app instance
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
