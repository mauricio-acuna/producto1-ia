"""
Laboratorio 2: Cost & Latency Monitoring - Sistema de Monitoreo de Performance
==============================================================================

Este laboratorio implementa un sistema completo de monitoreo de costes y latencia
para sistemas de IA, incluyendo tracking en tiempo real, alertas y dashboards.

Autor: Sistema de IA Educativo
Módulo: D - Métricas y Evaluación
"""

import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import uuid
import math

# =============================================================================
# 1. MODELOS DE DATOS
# =============================================================================

@dataclass
class CostMetrics:
    """Métricas de coste para llamadas a LLM"""
    request_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_input_usd: float
    cost_output_usd: float
    total_cost_usd: float
    timestamp: float
    operation_type: str = "completion"
    
    def cost_per_1k_tokens(self) -> float:
        """Coste por 1000 tokens"""
        return (self.total_cost_usd / self.total_tokens) * 1000 if self.total_tokens > 0 else 0
    
    def efficiency_score(self) -> float:
        """Score de eficiencia (output/input ratio)"""
        return self.output_tokens / self.input_tokens if self.input_tokens > 0 else 0

@dataclass
class LatencyMetrics:
    """Métricas de latencia para una request"""
    request_id: str
    total_time: float
    ttfb: float  # Time to First Byte
    tokens_generated: int
    tokens_per_second: float
    timestamp: float
    operation_type: str = "completion"
    
    def throughput_score(self) -> float:
        """Score de throughput basado en tokens/segundo"""
        if self.tokens_per_second <= 10:
            return 0.3
        elif self.tokens_per_second <= 25:
            return 0.6
        elif self.tokens_per_second <= 50:
            return 0.8
        else:
            return 1.0

@dataclass
class PerformanceAlert:
    """Alerta de performance"""
    alert_id: str
    alert_type: str  # "cost_spike", "latency_spike", "error_rate"
    severity: str    # "low", "medium", "high", "critical"
    message: str
    current_value: float
    threshold_value: float
    timestamp: float
    resolved: bool = False

@dataclass
class AggregatedMetrics:
    """Métricas agregadas para un período"""
    period_start: float
    period_end: float
    total_requests: int
    total_cost_usd: float
    avg_cost_per_request: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_tokens_per_second: float
    error_count: int
    error_rate: float

# =============================================================================
# 2. CALCULADORA DE COSTES
# =============================================================================

class CostCalculator:
    """Calculadora de costes para diferentes modelos de LLM"""
    
    # Precios actualizados por 1K tokens (Agosto 2025)
    MODEL_PRICING = {
        # OpenAI
        "gpt-4o": {"input": 0.0025, "output": 0.010},
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        
        # Anthropic Claude
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        
        # Google Gemini
        "gemini-1.5-flash": {"input": 0.00007, "output": 0.00021},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
        
        # Otros modelos
        "llama-3-70b": {"input": 0.0009, "output": 0.0009},
        "mixtral-8x7b": {"input": 0.0007, "output": 0.0007}
    }
    
    def __init__(self):
        self.custom_pricing = {}
    
    def add_custom_model(self, model_name: str, input_price: float, output_price: float):
        """Agregar modelo personalizado con precios"""
        self.custom_pricing[model_name] = {
            "input": input_price,
            "output": output_price
        }
    
    def calculate_cost(self, 
                      model_name: str,
                      input_tokens: int,
                      output_tokens: int,
                      request_id: str = None,
                      operation_type: str = "completion") -> CostMetrics:
        """Calcular coste de una llamada"""
        
        # Buscar en custom pricing primero
        if model_name in self.custom_pricing:
            pricing = self.custom_pricing[model_name]
        elif model_name in self.MODEL_PRICING:
            pricing = self.MODEL_PRICING[model_name]
        else:
            # Precio por defecto para modelos desconocidos
            pricing = {"input": 0.001, "output": 0.003}
        
        cost_input = (input_tokens / 1000) * pricing["input"]
        cost_output = (output_tokens / 1000) * pricing["output"]
        total_cost = cost_input + cost_output
        total_tokens = input_tokens + output_tokens
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        return CostMetrics(
            request_id=request_id,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_input_usd=cost_input,
            cost_output_usd=cost_output,
            total_cost_usd=total_cost,
            timestamp=time.time(),
            operation_type=operation_type
        )
    
    def estimate_monthly_cost(self, 
                            daily_requests: int,
                            avg_input_tokens: int,
                            avg_output_tokens: int,
                            model_name: str) -> Dict[str, float]:
        """Estimar coste mensual"""
        
        single_cost = self.calculate_cost(model_name, avg_input_tokens, avg_output_tokens)
        
        daily_cost = daily_requests * single_cost.total_cost_usd
        monthly_cost = daily_cost * 30
        annual_cost = daily_cost * 365
        
        return {
            "cost_per_request": single_cost.total_cost_usd,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "annual_cost": annual_cost,
            "requests_per_dollar": 1.0 / single_cost.total_cost_usd if single_cost.total_cost_usd > 0 else 0,
            "tokens_per_dollar": single_cost.total_tokens / single_cost.total_cost_usd if single_cost.total_cost_usd > 0 else 0
        }
    
    def compare_models(self, 
                      models: List[str],
                      avg_input_tokens: int,
                      avg_output_tokens: int) -> Dict[str, Dict[str, float]]:
        """Comparar costes entre diferentes modelos"""
        
        comparison = {}
        
        for model in models:
            try:
                cost_metrics = self.calculate_cost(model, avg_input_tokens, avg_output_tokens)
                comparison[model] = {
                    "total_cost": cost_metrics.total_cost_usd,
                    "cost_per_1k_tokens": cost_metrics.cost_per_1k_tokens(),
                    "input_cost": cost_metrics.cost_input_usd,
                    "output_cost": cost_metrics.cost_output_usd
                }
            except Exception as e:
                comparison[model] = {"error": str(e)}
        
        return comparison

# =============================================================================
# 3. TRACKER DE LATENCIA
# =============================================================================

class LatencyTracker:
    """Tracker de latencia con métricas detalladas"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.latencies = deque(maxlen=max_history)
        self.ttfb_times = deque(maxlen=max_history)
        self.active_requests = {}
        self.lock = threading.Lock()
    
    def start_request(self, request_id: str = None) -> str:
        """Iniciar tracking de una request"""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        with self.lock:
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "ttfb": None,
                "tokens_generated": 0
            }
        
        return request_id
    
    def record_ttfb(self, request_id: str):
        """Registrar Time to First Byte"""
        with self.lock:
            if request_id in self.active_requests:
                start_time = self.active_requests[request_id]["start_time"]
                ttfb = time.time() - start_time
                self.active_requests[request_id]["ttfb"] = ttfb
                self.ttfb_times.append(ttfb)
    
    def record_token(self, request_id: str):
        """Registrar generación de un token"""
        with self.lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]["tokens_generated"] += 1
    
    def end_request(self, request_id: str, operation_type: str = "completion") -> LatencyMetrics:
        """Finalizar tracking y retornar métricas"""
        with self.lock:
            if request_id not in self.active_requests:
                raise ValueError(f"Request {request_id} not found")
            
            request_data = self.active_requests[request_id]
            end_time = time.time()
            total_time = end_time - request_data["start_time"]
            
            tokens_generated = request_data["tokens_generated"]
            tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
            ttfb = request_data["ttfb"] or total_time
            
            metrics = LatencyMetrics(
                request_id=request_id,
                total_time=total_time,
                ttfb=ttfb,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                timestamp=end_time,
                operation_type=operation_type
            )
            
            self.latencies.append(metrics)
            del self.active_requests[request_id]
            
            return metrics
    
    def get_aggregated_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Obtener métricas agregadas"""
        with self.lock:
            if not self.latencies:
                return self._empty_metrics()
            
            recent_latencies = list(self.latencies)
            if last_n:
                recent_latencies = recent_latencies[-last_n:]
            
            total_times = [m.total_time for m in recent_latencies]
            ttfb_times = [m.ttfb for m in recent_latencies]
            tokens_per_sec = [m.tokens_per_second for m in recent_latencies if m.tokens_per_second > 0]
            
            return {
                "count": len(recent_latencies),
                "avg_total_time": statistics.mean(total_times),
                "p50_latency": statistics.median(total_times),
                "p95_latency": self._percentile(total_times, 0.95),
                "p99_latency": self._percentile(total_times, 0.99),
                "max_latency": max(total_times),
                "min_latency": min(total_times),
                "avg_ttfb": statistics.mean(ttfb_times) if ttfb_times else 0,
                "avg_tokens_per_second": statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
                "median_tokens_per_second": statistics.median(tokens_per_sec) if tokens_per_sec else 0
            }
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calcular percentil"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(p * (len(sorted_data) - 1))
        return sorted_data[index]
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Métricas vacías"""
        return {
            "count": 0,
            "avg_total_time": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "max_latency": 0.0,
            "min_latency": 0.0,
            "avg_ttfb": 0.0,
            "avg_tokens_per_second": 0.0,
            "median_tokens_per_second": 0.0
        }

# =============================================================================
# 4. SISTEMA DE ALERTAS
# =============================================================================

class AlertingSystem:
    """Sistema de alertas para métricas de performance"""
    
    def __init__(self):
        self.thresholds = {
            "cost_spike_multiplier": 2.0,      # 2x aumento de coste
            "latency_spike_multiplier": 2.0,   # 2x aumento de latencia
            "error_rate_threshold": 0.05,      # 5% tasa de error
            "cost_daily_limit": 100.0,         # $100 límite diario
            "latency_p95_threshold": 10.0,     # 10s P95 latency
            "tokens_per_second_min": 5.0       # Mínimo 5 tokens/segundo
        }
        
        self.alerts = []
        self.alert_callbacks = []
        self.baseline_metrics = {}
        
    def set_threshold(self, metric_name: str, value: float):
        """Configurar threshold para una métrica"""
        self.thresholds[metric_name] = value
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Agregar callback para manejar alertas"""
        self.alert_callbacks.append(callback)
    
    def update_baseline(self, metric_name: str, value: float):
        """Actualizar valor baseline para una métrica"""
        self.baseline_metrics[metric_name] = value
    
    def check_cost_metrics(self, cost_metrics: CostMetrics):
        """Verificar métricas de coste"""
        alerts = []
        
        # Check daily cost limit
        if cost_metrics.total_cost_usd > self.thresholds["cost_daily_limit"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="cost_daily_limit",
                severity="high",
                message=f"Daily cost limit exceeded: ${cost_metrics.total_cost_usd:.4f}",
                current_value=cost_metrics.total_cost_usd,
                threshold_value=self.thresholds["cost_daily_limit"],
                timestamp=time.time()
            )
            alerts.append(alert)
        
        # Check cost spike vs baseline
        if "avg_cost" in self.baseline_metrics:
            baseline_cost = self.baseline_metrics["avg_cost"]
            if cost_metrics.total_cost_usd > baseline_cost * self.thresholds["cost_spike_multiplier"]:
                alert = PerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="cost_spike",
                    severity="medium",
                    message=f"Cost spike detected: ${cost_metrics.total_cost_usd:.4f} vs baseline ${baseline_cost:.4f}",
                    current_value=cost_metrics.total_cost_usd,
                    threshold_value=baseline_cost * self.thresholds["cost_spike_multiplier"],
                    timestamp=time.time()
                )
                alerts.append(alert)
        
        self._process_alerts(alerts)
        return alerts
    
    def check_latency_metrics(self, latency_metrics: LatencyMetrics):
        """Verificar métricas de latencia"""
        alerts = []
        
        # Check latency spike vs baseline
        if "avg_latency" in self.baseline_metrics:
            baseline_latency = self.baseline_metrics["avg_latency"]
            if latency_metrics.total_time > baseline_latency * self.thresholds["latency_spike_multiplier"]:
                alert = PerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="latency_spike",
                    severity="medium",
                    message=f"Latency spike: {latency_metrics.total_time:.3f}s vs baseline {baseline_latency:.3f}s",
                    current_value=latency_metrics.total_time,
                    threshold_value=baseline_latency * self.thresholds["latency_spike_multiplier"],
                    timestamp=time.time()
                )
                alerts.append(alert)
        
        # Check low throughput
        if latency_metrics.tokens_per_second < self.thresholds["tokens_per_second_min"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="low_throughput",
                severity="low",
                message=f"Low throughput: {latency_metrics.tokens_per_second:.2f} tokens/s",
                current_value=latency_metrics.tokens_per_second,
                threshold_value=self.thresholds["tokens_per_second_min"],
                timestamp=time.time()
            )
            alerts.append(alert)
        
        self._process_alerts(alerts)
        return alerts
    
    def check_aggregated_metrics(self, metrics: AggregatedMetrics):
        """Verificar métricas agregadas"""
        alerts = []
        
        # Check error rate
        if metrics.error_rate > self.thresholds["error_rate_threshold"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="high_error_rate",
                severity="high",
                message=f"High error rate: {metrics.error_rate:.1%}",
                current_value=metrics.error_rate,
                threshold_value=self.thresholds["error_rate_threshold"],
                timestamp=time.time()
            )
            alerts.append(alert)
        
        # Check P95 latency
        if metrics.p95_latency > self.thresholds["latency_p95_threshold"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="high_p95_latency",
                severity="medium",
                message=f"High P95 latency: {metrics.p95_latency:.3f}s",
                current_value=metrics.p95_latency,
                threshold_value=self.thresholds["latency_p95_threshold"],
                timestamp=time.time()
            )
            alerts.append(alert)
        
        self._process_alerts(alerts)
        return alerts
    
    def _process_alerts(self, alerts: List[PerformanceAlert]):
        """Procesar alertas generadas"""
        for alert in alerts:
            self.alerts.append(alert)
            
            # Ejecutar callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
    
    def get_active_alerts(self, last_hours: int = 24) -> List[PerformanceAlert]:
        """Obtener alertas activas"""
        cutoff_time = time.time() - (last_hours * 3600)
        return [alert for alert in self.alerts 
                if alert.timestamp > cutoff_time and not alert.resolved]

# =============================================================================
# 5. MONITOR DE PERFORMANCE COMPLETO
# =============================================================================

class PerformanceMonitor:
    """Monitor completo de performance que integra coste y latencia"""
    
    def __init__(self, max_history: int = 1000):
        self.cost_calculator = CostCalculator()
        self.latency_tracker = LatencyTracker(max_history)
        self.alerting_system = AlertingSystem()
        
        self.cost_history = deque(maxlen=max_history)
        self.error_count = 0
        self.total_requests = 0
        
        self.lock = threading.Lock()
        
        # Setup default alert callback
        self.alerting_system.add_alert_callback(self._default_alert_handler)
    
    def start_monitoring_request(self, 
                                model_name: str,
                                operation_type: str = "completion") -> str:
        """Iniciar monitoreo de una request"""
        request_id = self.latency_tracker.start_request()
        
        with self.lock:
            self.total_requests += 1
        
        return request_id
    
    def record_ttfb(self, request_id: str):
        """Registrar Time to First Byte"""
        self.latency_tracker.record_ttfb(request_id)
    
    def record_token_generated(self, request_id: str):
        """Registrar generación de token"""
        self.latency_tracker.record_token(request_id)
    
    def end_monitoring_request(self, 
                             request_id: str,
                             model_name: str,
                             input_tokens: int,
                             output_tokens: int,
                             success: bool = True,
                             operation_type: str = "completion") -> Dict[str, Any]:
        """Finalizar monitoreo y calcular métricas"""
        
        # Obtener métricas de latencia
        latency_metrics = self.latency_tracker.end_request(request_id, operation_type)
        
        # Calcular métricas de coste
        cost_metrics = self.cost_calculator.calculate_cost(
            model_name, input_tokens, output_tokens, request_id, operation_type
        )
        
        # Actualizar historial
        with self.lock:
            self.cost_history.append(cost_metrics)
            if not success:
                self.error_count += 1
        
        # Verificar alertas
        cost_alerts = self.alerting_system.check_cost_metrics(cost_metrics)
        latency_alerts = self.alerting_system.check_latency_metrics(latency_metrics)
        
        return {
            "request_id": request_id,
            "cost_metrics": cost_metrics,
            "latency_metrics": latency_metrics,
            "alerts": cost_alerts + latency_alerts,
            "success": success
        }
    
    def record_error(self, error_type: str = "general"):
        """Registrar un error"""
        with self.lock:
            self.error_count += 1
    
    def get_dashboard_data(self, last_hours: int = 24) -> Dict[str, Any]:
        """Obtener datos para dashboard"""
        
        cutoff_time = time.time() - (last_hours * 3600)
        
        with self.lock:
            # Filtrar datos recientes
            recent_costs = [c for c in self.cost_history if c.timestamp > cutoff_time]
            
            # Métricas de coste
            total_cost = sum(c.total_cost_usd for c in recent_costs)
            avg_cost = statistics.mean([c.total_cost_usd for c in recent_costs]) if recent_costs else 0
            
            # Métricas de latencia
            latency_metrics = self.latency_tracker.get_aggregated_metrics()
            
            # Error rate
            error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
            
            # Alertas activas
            active_alerts = self.alerting_system.get_active_alerts(last_hours)
            
            return {
                "overview": {
                    "total_requests": self.total_requests,
                    "success_rate": 1.0 - error_rate,
                    "error_rate": error_rate,
                    "total_cost_usd": total_cost,
                    "avg_cost_per_request": avg_cost,
                    "period_hours": last_hours
                },
                "cost_metrics": {
                    "total_cost": total_cost,
                    "avg_cost": avg_cost,
                    "cost_by_model": self._group_costs_by_model(recent_costs),
                    "hourly_cost": self._calculate_hourly_costs(recent_costs, last_hours)
                },
                "latency_metrics": latency_metrics,
                "alerts": {
                    "active_count": len(active_alerts),
                    "alerts": active_alerts[:10],  # Mostrar solo las 10 más recientes
                    "by_severity": self._group_alerts_by_severity(active_alerts)
                },
                "trends": {
                    "cost_trend": self._calculate_cost_trend(recent_costs),
                    "latency_trend": self._calculate_latency_trend()
                }
            }
    
    def _group_costs_by_model(self, costs: List[CostMetrics]) -> Dict[str, Dict[str, float]]:
        """Agrupar costes por modelo"""
        model_costs = defaultdict(list)
        
        for cost in costs:
            model_costs[cost.model_name].append(cost.total_cost_usd)
        
        return {
            model: {
                "total_cost": sum(costs),
                "avg_cost": statistics.mean(costs),
                "request_count": len(costs)
            }
            for model, costs in model_costs.items()
        }
    
    def _calculate_hourly_costs(self, costs: List[CostMetrics], hours: int) -> List[float]:
        """Calcular costes por hora"""
        if not costs:
            return [0.0] * hours
        
        current_time = time.time()
        hourly_costs = [0.0] * hours
        
        for cost in costs:
            hour_index = int((current_time - cost.timestamp) / 3600)
            if 0 <= hour_index < hours:
                hourly_costs[-(hour_index + 1)] += cost.total_cost_usd
        
        return hourly_costs
    
    def _group_alerts_by_severity(self, alerts: List[PerformanceAlert]) -> Dict[str, int]:
        """Agrupar alertas por severidad"""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for alert in alerts:
            severity_counts[alert.severity] += 1
        
        return severity_counts
    
    def _calculate_cost_trend(self, costs: List[CostMetrics]) -> str:
        """Calcular tendencia de coste"""
        if len(costs) < 10:
            return "insufficient_data"
        
        # Comparar primera y segunda mitad
        mid_point = len(costs) // 2
        first_half_avg = statistics.mean([c.total_cost_usd for c in costs[:mid_point]])
        second_half_avg = statistics.mean([c.total_cost_usd for c in costs[mid_point:]])
        
        if second_half_avg > first_half_avg * 1.1:
            return "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_latency_trend(self) -> str:
        """Calcular tendencia de latencia"""
        latencies = list(self.latency_tracker.latencies)
        
        if len(latencies) < 10:
            return "insufficient_data"
        
        # Comparar primera y segunda mitad
        mid_point = len(latencies) // 2
        first_half_avg = statistics.mean([l.total_time for l in latencies[:mid_point]])
        second_half_avg = statistics.mean([l.total_time for l in latencies[mid_point:]])
        
        if second_half_avg > first_half_avg * 1.1:
            return "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _default_alert_handler(self, alert: PerformanceAlert):
        """Handler por defecto para alertas"""
        severity_icons = {
            "low": "💙",
            "medium": "🧡", 
            "high": "❤️",
            "critical": "🚨"
        }
        
        icon = severity_icons.get(alert.severity, "⚠️")
        print(f"{icon} ALERT [{alert.severity.upper()}]: {alert.message}")
    
    def export_metrics(self, filename: str, last_hours: int = 24):
        """Exportar métricas a archivo JSON"""
        dashboard_data = self.get_dashboard_data(last_hours)
        
        # Hacer serializable
        serializable_data = self._make_serializable(dashboard_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Metrics exported to {filename}")
    
    def _make_serializable(self, obj):
        """Convertir objetos a formato serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (CostMetrics, LatencyMetrics, PerformanceAlert)):
            return asdict(obj)
        else:
            return obj

# =============================================================================
# 6. SIMULADOR DE REQUESTS
# =============================================================================

class RequestSimulator:
    """Simulador de requests para testing del monitor"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def simulate_llm_request(self, 
                           model_name: str,
                           input_tokens: int,
                           output_tokens: int,
                           simulate_latency: bool = True,
                           success_rate: float = 0.95) -> Dict[str, Any]:
        """Simular una request a LLM"""
        
        # Iniciar monitoreo
        request_id = self.monitor.start_monitoring_request(model_name)
        
        if simulate_latency:
            # Simular TTFB
            ttfb_delay = 0.1 + (input_tokens / 10000)  # Delay basado en input size
            time.sleep(ttfb_delay)
            self.monitor.record_ttfb(request_id)
            
            # Simular generación de tokens
            token_delay = 0.02  # 50 tokens/segundo
            for _ in range(output_tokens):
                time.sleep(token_delay)
                self.monitor.record_token_generated(request_id)
        
        # Determinar éxito
        success = statistics.random() < success_rate
        
        if not success:
            self.monitor.record_error()
        
        # Finalizar monitoreo
        return self.monitor.end_monitoring_request(
            request_id, model_name, input_tokens, output_tokens, success
        )
    
    def simulate_batch_requests(self, 
                              scenarios: List[Dict[str, Any]],
                              delay_between: float = 0.1) -> List[Dict[str, Any]]:
        """Simular múltiples requests"""
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            print(f"🔄 Simulating request {i+1}/{len(scenarios)}: {scenario['model_name']}")
            
            result = self.simulate_llm_request(
                model_name=scenario.get('model_name', 'gpt-4o-mini'),
                input_tokens=scenario.get('input_tokens', 100),
                output_tokens=scenario.get('output_tokens', 50),
                simulate_latency=scenario.get('simulate_latency', True),
                success_rate=scenario.get('success_rate', 0.95)
            )
            
            results.append(result)
            
            if delay_between > 0:
                time.sleep(delay_between)
        
        return results

# =============================================================================
# 7. FUNCIÓN PRINCIPAL Y DEMO
# =============================================================================

def create_demo_scenarios() -> List[Dict[str, Any]]:
    """Crear scenarios de demo para testing"""
    
    return [
        {"model_name": "gpt-4o-mini", "input_tokens": 50, "output_tokens": 100},
        {"model_name": "gpt-4o", "input_tokens": 200, "output_tokens": 150},
        {"model_name": "claude-3-haiku", "input_tokens": 75, "output_tokens": 80},
        {"model_name": "gemini-1.5-flash", "input_tokens": 120, "output_tokens": 200},
        {"model_name": "gpt-4o-mini", "input_tokens": 300, "output_tokens": 250},
        {"model_name": "claude-3-sonnet", "input_tokens": 180, "output_tokens": 120},
        {"model_name": "gpt-4o", "input_tokens": 400, "output_tokens": 300, "success_rate": 0.8},  # Más errores
        {"model_name": "gemini-1.5-pro", "input_tokens": 250, "output_tokens": 180},
        {"model_name": "gpt-4o-mini", "input_tokens": 100, "output_tokens": 80},
        {"model_name": "claude-3-haiku", "input_tokens": 150, "output_tokens": 100}
    ]

def demo_cost_monitoring():
    """Demostración del sistema de monitoreo de costes"""
    
    print("💰 Cost & Latency Monitoring - Demo")
    print("=" * 50)
    
    # Crear monitor
    monitor = PerformanceMonitor()
    
    # Configurar thresholds custom
    monitor.alerting_system.set_threshold("cost_daily_limit", 1.0)  # $1 para testing
    monitor.alerting_system.set_threshold("latency_spike_multiplier", 1.5)
    
    # Establecer baselines
    monitor.alerting_system.update_baseline("avg_cost", 0.01)
    monitor.alerting_system.update_baseline("avg_latency", 1.0)
    
    # Crear simulador
    simulator = RequestSimulator(monitor)
    
    # Ejecutar scenarios
    scenarios = create_demo_scenarios()
    print(f"🚀 Running {len(scenarios)} simulated requests...")
    
    results = simulator.simulate_batch_requests(scenarios, delay_between=0.05)
    
    print(f"\n✅ Completed {len(results)} requests")
    
    # Obtener dashboard data
    dashboard = monitor.get_dashboard_data(last_hours=1)
    
    # Mostrar resumen
    print("\n📊 Performance Summary:")
    print("=" * 30)
    
    overview = dashboard["overview"]
    print(f"Total Requests: {overview['total_requests']}")
    print(f"Success Rate: {overview['success_rate']:.1%}")
    print(f"Total Cost: ${overview['total_cost_usd']:.4f}")
    print(f"Avg Cost/Request: ${overview['avg_cost_per_request']:.4f}")
    
    latency = dashboard["latency_metrics"]
    print(f"\nLatency Metrics:")
    print(f"  P50: {latency['p50_latency']:.3f}s")
    print(f"  P95: {latency['p95_latency']:.3f}s")
    print(f"  Avg Tokens/s: {latency['avg_tokens_per_second']:.1f}")
    
    # Mostrar costes por modelo
    print(f"\n💸 Cost by Model:")
    for model, cost_data in dashboard["cost_metrics"]["cost_by_model"].items():
        print(f"  {model}: ${cost_data['total_cost']:.4f} "
              f"(avg: ${cost_data['avg_cost']:.4f}, "
              f"requests: {cost_data['request_count']})")
    
    # Mostrar alertas
    alerts = dashboard["alerts"]
    if alerts["active_count"] > 0:
        print(f"\n🚨 Active Alerts ({alerts['active_count']}):")
        for alert in alerts["alerts"][:5]:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
    else:
        print(f"\n✅ No active alerts")
    
    # Exportar métricas
    monitor.export_metrics("performance_metrics.json")
    
    return monitor, dashboard

def demo_cost_calculator():
    """Demo de calculadora de costes"""
    
    print("\n💰 Cost Calculator Demo")
    print("=" * 30)
    
    calc = CostCalculator()
    
    # Comparar modelos
    models = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku", "gemini-1.5-flash"]
    comparison = calc.compare_models(models, avg_input_tokens=200, avg_output_tokens=100)
    
    print("🔍 Model Cost Comparison (200 input + 100 output tokens):")
    for model, costs in comparison.items():
        if "error" not in costs:
            print(f"  {model}: ${costs['total_cost']:.4f} "
                  f"(${costs['cost_per_1k_tokens']:.3f}/1K tokens)")
    
    # Estimación mensual
    print(f"\n📅 Monthly Cost Estimation (100 requests/day):")
    for model in models[:3]:  # Solo algunos modelos
        estimation = calc.estimate_monthly_cost(
            daily_requests=100,
            avg_input_tokens=200,
            avg_output_tokens=100,
            model_name=model
        )
        print(f"  {model}: ${estimation['monthly_cost']:.2f}/month "
              f"(${estimation['cost_per_request']:.4f}/request)")

if __name__ == "__main__":
    # Ejecutar demo completo
    print("🚀 Starting Cost & Latency Monitoring Demo\n")
    
    # Demo calculadora
    demo_cost_calculator()
    
    # Demo monitor completo
    monitor, dashboard = demo_cost_monitoring()
    
    print("\n🎉 Cost & Latency Monitoring Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Real-time cost calculation for multiple LLM models")
    print("✅ Latency tracking with percentiles and throughput")
    print("✅ Automated alerting system with configurable thresholds")
    print("✅ Performance dashboard with trends and summaries")
    print("✅ JSON export for integration with external systems")
    print("✅ Request simulation for testing and validation")
    print("\n📊 Ready for production monitoring!")
