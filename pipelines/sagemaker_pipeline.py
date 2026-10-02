"""Build and optionally run the SageMaker Phase 6 pipeline.

The default command only renders the pipeline definition.  Use ``--upsert``
to create/update it and ``--start`` to start an execution explicitly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingOutput
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.mlops.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.mlops.workflow.functions import JsonGet
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.steps import ConditionStep, ProcessingStep


ROOT = Path(__file__).resolve().parents[1]
PROCESSING_SCRIPT = ROOT / "src" / "processing" / "preprocess.py"
EVALUATION_SCRIPT = ROOT / "src" / "evaluation" / "pipeline_evaluate.py"


def build_pipeline(
    *,
    role: str,
    bucket: str,
    region: str,
    pipeline_name: str,
    default_input: str,
) -> Pipeline:
    """Return a compileable pipeline definition without creating AWS resources."""
    session = PipelineSession(boto_session=boto3.Session(region_name=region))
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.large",
        image_scope="training",
    )

    input_data = ParameterString(
        name="InputDataUri", default_value=default_input
    )
    min_roc_auc = ParameterFloat(name="MinRocAuc", default_value=0.80)
    min_f1 = ParameterFloat(name="MinF1", default_value=0.75)

    processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
    )
    process_step = ProcessingStep(
        name="Preprocess",
        step_args=processor.run(
            code="src/processing/preprocess.py",
            source_dir=str(ROOT),
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{bucket}/phase6/{ExecutionVariables.PipelineExecutionId}/processed",
                )
            ],
            arguments=[
                "/opt/ml/processing/input/input.csv",
                "/opt/ml/processing/output",
            ],
        ),
    )

    evaluation_property = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="metrics.json",
    )
    evaluation_step = ProcessingStep(
        name="Evaluate",
        step_args=processor.run(
            code="src/evaluation/pipeline_evaluate.py",
            source_dir=str(ROOT),
            inputs=[
                ProcessingInput(
                    source=process_step.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"s3://{bucket}/phase6/{ExecutionVariables.PipelineExecutionId}/evaluation",
                )
            ],
            arguments=[
                "--train",
                "/opt/ml/processing/input/train.csv",
                "--test",
                "/opt/ml/processing/input/test.csv",
                "--model-dir",
                "/opt/ml/processing/evaluation/model",
                "--output-dir",
                "/opt/ml/processing/evaluation",
            ],
        ),
        property_files=[evaluation_property],
    )

    quality_step = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_property,
                    json_path="roc_auc",
                ),
                right=min_roc_auc,
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_property,
                    json_path="f1",
                ),
                right=min_f1,
            ),
        ],
        if_steps=[],
        else_steps=[],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[input_data, min_roc_auc, min_f1],
        steps=[process_step, evaluation_step, quality_step],
        sagemaker_session=session,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-s3-uri", required=True)
    parser.add_argument("--pipeline-name", default="customer-churn-phase6")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"))
    parser.add_argument("--role", default=None)
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if args.start and not args.upsert:
        parser.error("--start requires --upsert")

    role = args.role or get_execution_role()
    pipeline = build_pipeline(
        role=role,
        bucket=args.bucket,
        region=args.region,
        pipeline_name=args.pipeline_name,
        default_input=args.input_s3_uri,
    )
    definition = pipeline.definition()
    print(f"Pipeline: {args.pipeline_name}")
    print(f"Definition generated: {len(definition)} bytes")
    if args.upsert:
        pipeline.upsert(role_arn=role)
        print("Pipeline upserted.")
    if args.start:
        execution = pipeline.start()
        print(f"Execution ARN: {execution.arn}")


if __name__ == "__main__":
    main()
