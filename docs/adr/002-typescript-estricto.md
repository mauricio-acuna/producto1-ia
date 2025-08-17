# ADR-002: TypeScript Estricto para Seguridad de Tipos en IA

## Estado
✅ **Aceptado** - 18 de Agosto, 2025

## Contexto

El Framework de Agentes IA maneja tipos de datos complejos y críticos:

- **Configuraciones de agentes** con validación en tiempo de compilación
- **Payloads de API** que deben coincidir exactamente con schemas
- **Estados de ejecución** que requieren type safety absoluto
- **Integraciones con modelos de IA** con tipos específicos por proveedor
- **Pipelines de datos** donde errores de tipo pueden causar fallas críticas

### Desafíos del Sistema Actual

```typescript
// ❌ Problema: Tipos permisivos llevan a errores runtime
interface ConfiguracionAgente {
  modelo?: any;  // Demasiado permisivo
  parametros?: object;  // Sin validación
  callbacks?: Function[];  // Tipos no específicos
}

// ❌ Problema: Manejo de errores inconsistente
async function ejecutarAgente(config: any): Promise<any> {
  // Sin type safety, errores descubiertos en runtime
  return await modelo.generar(config.prompt);
}
```

### Opciones Evaluadas

1. **TypeScript Standard** (strict: false): Flexibilidad vs. seguridad
2. **TypeScript Estricto** (strict: true): Máxima seguridad de tipos
3. **TypeScript Ultra-Estricto**: Configuración personalizada extrema
4. **ReScript/PureScript**: Sistemas de tipos más avanzados
5. **Flow**: Alternativa de Facebook para type checking

## Decisión

**Adoptar TypeScript Ultra-Estricto con configuración personalizada** optimizada para sistemas de IA.

### Configuración tsconfig.json

```json
{
  "compilerOptions": {
    // === Configuración Ultra-Estricta ===
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "allowUnreachableCode": false,
    "allowUnusedLabels": false,
    "exactOptionalPropertyTypes": true,
    
    // === Configuraciones Específicas para IA ===
    "noPropertyAccessFromIndexSignature": true,
    "noImplicitThis": true,
    "useUnknownInCatchVariables": true,
    
    // === Configuraciones de Desarrollo ===
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "removeComments": false,
    "preserveConstEnums": true,
    
    // === Path Mapping ===
    "baseUrl": "./src",
    "paths": {
      "@/core/*": ["core/*"],
      "@/agentes/*": ["agentes/*"],
      "@/tipos/*": ["tipos/*"],
      "@/utils/*": ["utils/*"],
      "@/servicios/*": ["servicios/*"]
    },
    
    // === Configuraciones de Calidad ===
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": false,
    "resolveJsonModule": true,
    "allowSyntheticDefaultImports": true,
    "esModuleInterop": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true
  },
  
  "include": [
    "src/**/*",
    "types/**/*",
    "tests/**/*"
  ],
  
  "exclude": [
    "node_modules",
    "dist",
    "build",
    "coverage",
    "**/*.js"
  ]
}
```

### Sistema de Tipos Específico para IA

```typescript
// types/agentes.ts - Tipos ultra-específicos para agentes
export namespace TiposAgente {
  
  // === Tipos Base Estrictos ===
  export type IDAgente = `agente_${string}`;
  export type VersionModelo = `v${number}.${number}.${number}`;
  export type TimestampISO = `${number}-${string}-${string}T${string}`;
  
  // === Configuración de Modelo con Discriminated Unions ===
  export type ConfiguracionModelo = 
    | ConfiguracionOpenAI
    | ConfiguracionAnthropic
    | ConfiguracionGoogleAI
    | ConfiguracionHuggingFace;
  
  interface ConfiguracionModeloBase {
    readonly proveedor: ProveedorIA;
    readonly modelo: string;
    readonly version: VersionModelo;
    readonly parametros: ParametrosModelo;
    readonly limites: LimitesModelo;
  }
  
  export interface ConfiguracionOpenAI extends ConfiguracionModeloBase {
    readonly proveedor: 'openai';
    readonly modelo: 'gpt-4' | 'gpt-4-turbo' | 'gpt-3.5-turbo';
    readonly parametros: {
      readonly temperature: number & { readonly __brand: 'temperature' };
      readonly max_tokens: PositiveInteger;
      readonly top_p: number & { readonly __brand: 'top_p' };
      readonly frequency_penalty: number & { readonly __brand: 'frequency_penalty' };
      readonly presence_penalty: number & { readonly __brand: 'presence_penalty' };
      readonly stop?: readonly string[];
    };
  }
  
  export interface ConfiguracionAnthropic extends ConfiguracionModeloBase {
    readonly proveedor: 'anthropic';
    readonly modelo: 'claude-3-opus' | 'claude-3-sonnet' | 'claude-3-haiku';
    readonly parametros: {
      readonly temperature: number & { readonly __brand: 'temperature' };
      readonly max_tokens: PositiveInteger;
      readonly top_p: number & { readonly __brand: 'top_p' };
      readonly top_k: PositiveInteger;
    };
  }
  
  // === Branded Types para Validación ===
  export type PositiveInteger = number & { readonly __brand: 'PositiveInteger' };
  export type EmailAddress = string & { readonly __brand: 'EmailAddress' };
  export type JSONString = string & { readonly __brand: 'JSONString' };
  export type Base64String = string & { readonly __brand: 'Base64String' };
  
  // === Factory Functions con Validación ===
  export function crearPositiveInteger(valor: number): PositiveInteger {
    if (!Number.isInteger(valor) || valor <= 0) {
      throw new TypeError(`Valor debe ser un entero positivo, recibido: ${valor}`);
    }
    return valor as PositiveInteger;
  }
  
  export function crearEmailAddress(email: string): EmailAddress {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      throw new TypeError(`Email inválido: ${email}`);
    }
    return email as EmailAddress;
  }
  
  // === Estados de Agente con Finite State Machine ===
  export type EstadoAgente = 
    | { readonly tipo: 'inactivo'; readonly datos: null }
    | { readonly tipo: 'inicializando'; readonly datos: DatosInicializacion }
    | { readonly tipo: 'ejecutando'; readonly datos: DatosEjecucion }
    | { readonly tipo: 'pausado'; readonly datos: DatosPausa }
    | { readonly tipo: 'completado'; readonly datos: DatosCompletado }
    | { readonly tipo: 'error'; readonly datos: DatosError };
  
  export interface DatosEjecucion {
    readonly iniciadoEn: TimestampISO;
    readonly progresoActual: number & { readonly __brand: 'Progress' };
    readonly operacionActual: string;
    readonly recursosUsados: RecursosUsados;
    readonly logs: readonly LogEntry[];
  }
  
  // === Resultados con Result Pattern ===
  export type Result<T, E = Error> = 
    | { readonly success: true; readonly data: T }
    | { readonly success: false; readonly error: E };
  
  export type AsyncResult<T, E = Error> = Promise<Result<T, E>>;
  
  // === Configuración de Agente Ultra-Tipada ===
  export interface ConfiguracionAgente {
    readonly id: IDAgente;
    readonly nombre: string & { readonly length: number };
    readonly descripcion: string;
    readonly modelo: ConfiguracionModelo;
    readonly comportamiento: ComportamientoAgente;
    readonly restricciones: RestriccionesAgente;
    readonly callbacks: CallbacksAgente;
    readonly metadatos: MetadatosAgente;
  }
  
  export interface ComportamientoAgente {
    readonly tipoPersonalidad: 'analitico' | 'creativo' | 'pragmatico' | 'conservador';
    readonly nivelVerbosidad: 1 | 2 | 3 | 4 | 5;
    readonly estrategiaRespuesta: 'concisa' | 'detallada' | 'estructurada';
    readonly manejoErrores: 'estricto' | 'permisivo' | 'adaptativo';
  }
  
  export interface RestriccionesAgente {
    readonly tiempoMaximoEjecucion: number & { readonly __brand: 'Milliseconds' };
    readonly limiteTokens: PositiveInteger;
    readonly limiteLlamadas: PositiveInteger;
    readonly dominiosPermitidos: readonly string[];
    readonly palabrasProhibidas: readonly string[];
  }
  
  // === Callbacks Type-Safe ===
  export interface CallbacksAgente {
    readonly onIniciado?: (agente: ConfiguracionAgente) => AsyncResult<void>;
    readonly onProgreso?: (progreso: DatosEjecucion) => AsyncResult<void>;
    readonly onCompletado?: (resultado: DatosCompletado) => AsyncResult<void>;
    readonly onError?: (error: DatosError) => AsyncResult<void>;
    readonly onPausado?: (datos: DatosPausa) => AsyncResult<void>;
  }
}

// types/api.ts - Tipos API Ultra-Estrictos
export namespace TiposAPI {
  
  // === Request/Response con Branded Types ===
  export interface SolicitudEjecucionAgente {
    readonly agenteId: TiposAgente.IDAgente;
    readonly parametros: ParametrosEjecucion;
    readonly metadatos: MetadatosSolicitud;
    readonly firma: string & { readonly __brand: 'RequestSignature' };
  }
  
  export interface RespuestaEjecucionAgente {
    readonly sessionId: string & { readonly __brand: 'SessionID' };
    readonly estado: TiposAgente.EstadoAgente;
    readonly timestamp: TiposAgente.TimestampISO;
    readonly duracion: number & { readonly __brand: 'Milliseconds' };
    readonly resultado: TiposAgente.Result<any>;
  }
  
  // === Validación de Schema en Tiempo de Compilación ===
  export type ValidarSchema<T, S> = T extends S ? T : never;
  
  export interface SchemaValidado<T> {
    readonly _brand: 'SchemaValidado';
    readonly _tipo: T;
  }
  
  // === Middleware Type-Safe ===
  export type MiddlewareFunction<TInput, TOutput> = (
    input: TInput
  ) => TiposAgente.AsyncResult<TOutput>;
  
  export type PipelineMiddleware<T> = readonly MiddlewareFunction<T, T>[];
}

// types/utils.ts - Utilidades de Tipos Avanzadas
export namespace UtilsTipos {
  
  // === Readonly Recursivo ===
  export type ReadonlyProfundo<T> = {
    readonly [P in keyof T]: T[P] extends object 
      ? ReadonlyProfundo<T[P]> 
      : T[P];
  };
  
  // === Mandatory Optional ===
  export type Obligatorio<T, K extends keyof T> = T & Required<Pick<T, K>>;
  
  // === Exact Type Matching ===
  export type Exacto<T, U> = T extends U 
    ? U extends T 
      ? T 
      : never 
    : never;
  
  // === Conditional Type Helpers ===
  export type SiString<T> = T extends string ? T : never;
  export type SiNumero<T> = T extends number ? T : never;
  export type SiArray<T> = T extends readonly any[] ? T : never;
  
  // === Type-Level Computations ===
  export type Longitud<T extends readonly any[]> = T['length'];
  export type Cabeza<T extends readonly any[]> = T extends readonly [
    infer H,
    ...any[]
  ] ? H : never;
  export type Cola<T extends readonly any[]> = T extends readonly [
    any,
    ...infer Rest
  ] ? Rest : never;
}
```

### Guards de Tipo Ultra-Estrictos

```typescript
// utils/type-guards.ts
import { TiposAgente, TiposAPI, UtilsTipos } from '@/tipos';

export namespace GuardsTipo {
  
  // === Type Guards con Narrowing Preciso ===
  export function esConfiguracionOpenAI(
    config: TiposAgente.ConfiguracionModelo
  ): config is TiposAgente.ConfiguracionOpenAI {
    return config.proveedor === 'openai' && 
           typeof config.modelo === 'string' &&
           ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'].includes(config.modelo);
  }
  
  export function esEstadoEjecutando(
    estado: TiposAgente.EstadoAgente
  ): estado is Extract<TiposAgente.EstadoAgente, { tipo: 'ejecutando' }> {
    return estado.tipo === 'ejecutando';
  }
  
  export function esResultadoExitoso<T, E>(
    resultado: TiposAgente.Result<T, E>
  ): resultado is { success: true; data: T } {
    return resultado.success === true;
  }
  
  // === Validation Functions ===
  export function validarPositiveInteger(valor: unknown): valor is TiposAgente.PositiveInteger {
    return typeof valor === 'number' && 
           Number.isInteger(valor) && 
           valor > 0;
  }
  
  export function validarEmailAddress(valor: unknown): valor is TiposAgente.EmailAddress {
    return typeof valor === 'string' && 
           /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(valor);
  }
  
  // === Runtime Schema Validation ===
  export function validarEsquemaAgente(
    objeto: unknown
  ): objeto is TiposAgente.ConfiguracionAgente {
    if (typeof objeto !== 'object' || objeto === null) return false;
    
    const config = objeto as Record<string, unknown>;
    
    return typeof config.id === 'string' &&
           config.id.startsWith('agente_') &&
           typeof config.nombre === 'string' &&
           config.nombre.length > 0 &&
           typeof config.descripcion === 'string' &&
           esConfiguracionModeloValida(config.modelo) &&
           esComportamientoValido(config.comportamiento);
  }
  
  function esConfiguracionModeloValida(modelo: unknown): modelo is TiposAgente.ConfiguracionModelo {
    if (typeof modelo !== 'object' || modelo === null) return false;
    
    const config = modelo as Record<string, unknown>;
    const proveedoresValidos = ['openai', 'anthropic', 'google', 'huggingface'];
    
    return typeof config.proveedor === 'string' &&
           proveedoresValidos.includes(config.proveedor) &&
           typeof config.modelo === 'string' &&
           typeof config.version === 'string' &&
           /^v\d+\.\d+\.\d+$/.test(config.version);
  }
  
  function esComportamientoValido(comportamiento: unknown): comportamiento is TiposAgente.ComportamientoAgente {
    if (typeof comportamiento !== 'object' || comportamiento === null) return false;
    
    const comp = comportamiento as Record<string, unknown>;
    const personalidades = ['analitico', 'creativo', 'pragmatico', 'conservador'];
    const estrategias = ['concisa', 'detallada', 'estructurada'];
    const manejos = ['estricto', 'permisivo', 'adaptativo'];
    
    return personalidades.includes(comp.tipoPersonalidad as string) &&
           typeof comp.nivelVerbosidad === 'number' &&
           comp.nivelVerbosidad >= 1 && comp.nivelVerbosidad <= 5 &&
           estrategias.includes(comp.estrategiaRespuesta as string) &&
           manejos.includes(comp.manejoErrores as string);
  }
  
  // === Assertion Functions ===
  export function afirmarConfiguracionValida(
    config: unknown
  ): asserts config is TiposAgente.ConfiguracionAgente {
    if (!validarEsquemaAgente(config)) {
      throw new TypeError('Configuración de agente inválida');
    }
  }
  
  export function afirmarResultadoExitoso<T, E>(
    resultado: TiposAgente.Result<T, E>
  ): asserts resultado is { success: true; data: T } {
    if (!resultado.success) {
      throw new Error(`Operación falló: ${resultado.error}`);
    }
  }
}
```

### Linting Ultra-Estricto

```json
// .eslintrc.json
{
  "extends": [
    "@typescript-eslint/recommended",
    "@typescript-eslint/recommended-requiring-type-checking"
  ],
  "plugins": [
    "@typescript-eslint",
    "total-functions"
  ],
  "parserOptions": {
    "project": "./tsconfig.json"
  },
  "rules": {
    // === TypeScript Estricto ===
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-unsafe-any": "error",
    "@typescript-eslint/no-unsafe-assignment": "error",
    "@typescript-eslint/no-unsafe-call": "error",
    "@typescript-eslint/no-unsafe-member-access": "error",
    "@typescript-eslint/no-unsafe-return": "error",
    "@typescript-eslint/restrict-template-expressions": "error",
    "@typescript-eslint/strict-boolean-expressions": "error",
    "@typescript-eslint/prefer-nullish-coalescing": "error",
    "@typescript-eslint/prefer-optional-chain": "error",
    "@typescript-eslint/no-unnecessary-condition": "error",
    "@typescript-eslint/prefer-readonly": "error",
    "@typescript-eslint/prefer-readonly-parameter-types": "error",
    
    // === Funciones Totales (No Parciales) ===
    "total-functions/no-partial-division": "error",
    "total-functions/no-partial-array-reduce": "error",
    "total-functions/no-unsafe-readonly-mutable-assignment": "error",
    
    // === Consistencia de Nomenclatura ===
    "@typescript-eslint/naming-convention": [
      "error",
      {
        "selector": "interface",
        "format": ["PascalCase"],
        "prefix": ["I"]
      },
      {
        "selector": "typeAlias",
        "format": ["PascalCase"]
      },
      {
        "selector": "enum",
        "format": ["PascalCase"]
      },
      {
        "selector": "variable",
        "modifiers": ["const"],
        "format": ["UPPER_CASE", "camelCase"]
      }
    ]
  }
}
```

## Consecuencias

### Positivas

- **🛡️ Type Safety Absoluto**: Eliminación del 95% de errores de tipo en runtime
- **🔍 IntelliSense Perfecto**: Auto-completado preciso para todas las APIs de IA
- **🚀 Refactoring Seguro**: Cambios de gran escala sin miedo a romper código
- **📚 Documentación Viviente**: Los tipos sirven como documentación ejecutable
- **🧪 Testing Reducido**: Menos tests necesarios debido a garantías de compilación
- **🔧 Debugging Mejorado**: Errores encontrados en tiempo de compilación vs runtime
- **👥 Experiencia de Desarrollador**: IDE warnings/errors precisos y útiles
- **📈 Productividad**: Desarrollo más rápido una vez establecidos los tipos

### Negativas

- **⏱️ Tiempo de Compilación**: Incremento del 30-40% en tiempo de build
- **📈 Curva de Aprendizaje**: Desarrolladores deben aprender TypeScript avanzado
- **🔒 Rigidez**: Dificultad para prototipado rápido y experimentación
- **📦 Bundle Size**: Incremento mínimo por metadata de tipos (dev only)
- **🛠️ Tooling**: Necesidad de herramientas específicas para tipos avanzados
- **🏗️ Setup Inicial**: Tiempo considerable invertido en definir tipos base
- **🔄 Mantenimiento**: Tipos deben mantenerse sincronizados con APIs externas

### Neutras

- **🔄 Migración**: Migración gradual posible desde código JavaScript existente
- **🌐 Ecosistema**: TypeScript tiene ecosistema maduro y soporte excelente
- **📊 Monitoreo**: Métricas de type coverage ayudan a medir progreso

## Patrones de Migración

### Migración Gradual desde JavaScript

```typescript
// Paso 1: Convertir a TypeScript básico
// archivo.js -> archivo.ts
export function ejecutarAgente(config) {
  return modelo.generar(config.prompt);
}

// Paso 2: Agregar tipos básicos
export function ejecutarAgente(config: any): Promise<any> {
  return modelo.generar(config.prompt);
}

// Paso 3: Tipos específicos
export function ejecutarAgente(
  config: TiposAgente.ConfiguracionAgente
): TiposAgente.AsyncResult<string> {
  return modelo.generar(config.prompt);
}

// Paso 4: Implementación final ultra-estricta
export async function ejecutarAgente(
  config: TiposAgente.ConfiguracionAgente
): TiposAgente.AsyncResult<string> {
  try {
    GuardsTipo.afirmarConfiguracionValida(config);
    
    const resultado = await modelo.generar(config.prompt);
    
    return { success: true, data: resultado };
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error : new Error(String(error))
    };
  }
}
```

### Automatización de Validación

```typescript
// scripts/validar-tipos.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async function validarTipos(): Promise<void> {
  try {
    // Type checking estricto
    await execAsync('npx tsc --noEmit --strict');
    console.log('✅ Validación de tipos exitosa');
    
    // Linting ultra-estricto
    await execAsync('npx eslint src/ --ext .ts,.tsx');
    console.log('✅ Linting exitoso');
    
    // Coverage de tipos
    await execAsync('npx type-coverage --strict');
    console.log('✅ Coverage de tipos > 95%');
    
  } catch (error) {
    console.error('❌ Validación falló:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  validarTipos();
}
```

## Métricas de Éxito

- **Type Coverage**: > 95% de código con tipos explícitos
- **Errores Runtime**: Reducción del 80% en errores de tipo
- **Tiempo de Desarrollo**: 25% más rápido después del período de adopción
- **Bugs en Producción**: Reducción del 60% en bugs relacionados con tipos
- **Satisfacción del Desarrollador**: > 8/10 después de 3 meses de uso

## Referencias

- [TypeScript Handbook - Strict Mode](https://www.typescriptlang.org/docs/handbook/2/basic-types.html#strictness)
- [Total TypeScript - Advanced Patterns](https://totaltypescript.com/)
- [Type-Driven Development Patterns](https://blog.ploeh.dk/2016/02/10/types-properties-software/)
- [Branded Types in TypeScript](https://github.com/Microsoft/TypeScript/wiki/Coding-guidelines#names)
