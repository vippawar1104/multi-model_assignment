"""
Table processor module for the Multi-Modal RAG system.
Handles table extraction, processing, and conversion for structured data.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)


class TableProcessor:
    """
    Basic table processing for multi-modal RAG systems.
    """

    def __init__(
        self,
        preserve_structure: bool = True,
        max_columns: Optional[int] = None,
        convert_to_markdown: bool = True,
        extract_metadata: bool = True
    ):
        """
        Initialize table processor.

        Args:
            preserve_structure: Whether to preserve table structure
            max_columns: Maximum number of columns to process
            convert_to_markdown: Whether to convert tables to markdown
            extract_metadata: Whether to extract table metadata
        """
        self.preserve_structure = preserve_structure or get_config_value("preprocessing.table.preserve_structure", True)
        self.max_columns = max_columns or get_config_value("preprocessing.table.max_columns", 10)
        self.convert_to_markdown = convert_to_markdown or get_config_value("preprocessing.table.convert_to_markdown", True)
        self.extract_metadata = extract_metadata

        logger.info(f"TableProcessor initialized with max_columns={self.max_columns}")

    def process_table(
        self,
        table_data: Union[List[List[Any]], Dict[str, Any]],
        table_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single table.

        Args:
            table_data: Table data in various formats
            table_id: Optional table identifier
            metadata: Optional metadata

        Returns:
            Processed table dictionary
        """
        try:
            # Convert to list of lists format
            table_rows = self._convert_to_list_format(table_data)

            if not table_rows:
                logger.warning("Empty table provided")
                return self._create_empty_result(table_id, metadata)

            # Initialize result
            result = {
                "table_id": table_id or f"table_{hash(str(table_data))}",
                "original_shape": (len(table_rows), len(table_rows[0]) if table_rows else 0),
                "columns": [],
                "dtypes": {},
                "metadata": metadata or {}
            }

            # Clean and preprocess
            cleaned_rows = self._clean_table(table_rows)
            result["processed_shape"] = (len(cleaned_rows), len(cleaned_rows[0]) if cleaned_rows else 0)

            # Extract metadata
            if self.extract_metadata:
                table_metadata = self._extract_table_metadata(cleaned_rows)
                result["metadata"].update(table_metadata)

            # Convert to different formats
            result["dataframe"] = cleaned_rows

            if self.convert_to_markdown:
                result["markdown"] = self._convert_to_markdown(cleaned_rows)

            # Create text representation
            result["text_representation"] = self._create_text_representation(cleaned_rows)

            # Create chunks for embedding
            result["chunks"] = self._create_table_chunks(cleaned_rows, result)

            logger.info(f"Processed table {result['table_id']} with shape {result['processed_shape']}")
            return result

        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return {
                "table_id": table_id or "unknown",
                "error": str(e),
                "metadata": metadata or {}
            }

    def _convert_to_list_format(self, table_data: Union[List[List[Any]], Dict[str, Any]]) -> List[List[Any]]:
        """
        Convert various table formats to list of lists.

        Args:
            table_data: Input table data

        Returns:
            Table as list of lists
        """
        if isinstance(table_data, list):
            # Already list of lists
            return table_data

        elif isinstance(table_data, dict):
            # Dictionary format
            if "data" in table_data:
                return self._convert_to_list_format(table_data["data"])
            else:
                # Key-value pairs
                return [["Key", "Value"]] + [[k, v] for k, v in table_data.items()]

        else:
            # Try to convert to string and parse
            try:
                text = str(table_data)
                # Simple CSV-like parsing
                lines = text.strip().split('\n')
                if lines:
                    rows = [line.split(',') for line in lines]
                    return rows
            except:
                pass

        # Fallback
        return []

    def _clean_table(self, table_rows: List[List[Any]]) -> List[List[Any]]:
        """
        Clean and preprocess table data.

        Args:
            table_rows: Input table rows

        Returns:
            Cleaned table rows
        """
        if not table_rows:
            return table_rows

        # Remove completely empty rows
        cleaned_rows = [row for row in table_rows if any(str(cell).strip() for cell in row)]

        if not cleaned_rows:
            return []

        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in cleaned_rows)
        for row in cleaned_rows:
            while len(row) < max_cols:
                row.append("")

        # Limit columns if specified
        if self.max_columns and max_cols > self.max_columns:
            cleaned_rows = [row[:self.max_columns] for row in cleaned_rows]
            logger.warning(f"Truncated table to {self.max_columns} columns")

        # Clean cell values
        for row in cleaned_rows:
            for i, cell in enumerate(row):
                row[i] = self._clean_cell_value(cell)

        return cleaned_rows

    def _clean_cell_value(self, value: Any) -> str:
        """Clean individual cell value."""
        if value is None:
            return ""
        return str(value).strip()

    def _extract_table_metadata(self, table_rows: List[List[Any]]) -> Dict[str, Any]:
        """
        Extract metadata from table.

        Args:
            table_rows: Table rows

        Returns:
            Metadata dictionary
        """
        if not table_rows:
            return {}

        num_rows = len(table_rows)
        num_columns = len(table_rows[0]) if table_rows else 0

        # Check if first row looks like headers
        has_headers = self._has_headers(table_rows)

        metadata = {
            "num_rows": num_rows,
            "num_columns": num_columns,
            "has_headers": has_headers,
            "total_cells": num_rows * num_columns
        }

        return metadata

    def _has_headers(self, table_rows: List[List[Any]]) -> bool:
        """
        Determine if table has headers.

        Args:
            table_rows: Table rows

        Returns:
            True if headers detected
        """
        if len(table_rows) < 2:
            return True

        # Simple heuristic: if first row has different characteristics
        first_row = table_rows[0]
        second_row = table_rows[1] if len(table_rows) > 1 else []

        # Check if first row has more non-numeric values
        first_row_str_count = sum(1 for cell in first_row if not self._is_numeric(str(cell)))
        second_row_str_count = sum(1 for cell in second_row if not self._is_numeric(str(cell)))

        return first_row_str_count >= second_row_str_count

    def _is_numeric(self, value: str) -> bool:
        """Check if string represents a number."""
        try:
            float(value.replace(',', '').replace('%', ''))
            return True
        except ValueError:
            return False

    def _convert_to_markdown(self, table_rows: List[List[Any]]) -> str:
        """
        Convert table to markdown format.

        Args:
            table_rows: Table rows

        Returns:
            Markdown table string
        """
        if not table_rows:
            return ""

        lines = []

        # Add headers if present
        if self._has_headers(table_rows) and len(table_rows) > 1:
            header_line = " | ".join(str(cell) for cell in table_rows[0])
            lines.append(header_line)
            lines.append("-" * len(header_line))
            data_rows = table_rows[1:]
        else:
            data_rows = table_rows

        # Add data rows (limit to 10 for readability)
        for row in data_rows[:10]:
            row_line = " | ".join(str(cell) for cell in row)
            lines.append(row_line)

        if len(data_rows) > 10:
            lines.append(f"... ({len(data_rows) - 10} more rows)")

        return "\n".join(lines)

    def _create_text_representation(self, table_rows: List[List[Any]]) -> str:
        """
        Create text representation of table.

        Args:
            table_rows: Table rows

        Returns:
            Text representation
        """
        return self._convert_to_markdown(table_rows)

    def _create_table_chunks(self, table_rows: List[List[Any]], table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from table for embedding.

        Args:
            table_rows: Table rows
            table_info: Table information

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        # Create summary chunk
        summary_text = f"Table: {table_info['table_id']}\n"
        summary_text += f"Shape: {table_info['processed_shape']}\n"
        summary_text += f"Content preview:\n{table_info['text_representation']}"

        chunks.append({
            "id": f"{table_info['table_id']}_summary",
            "text": summary_text,
            "type": "table_summary",
            "table_id": table_info['table_id'],
            "chunk_index": 0,
            "total_chunks": 1
        })

        return chunks

    def _create_empty_result(self, table_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create result for empty table."""
        return {
            "table_id": table_id or "empty_table",
            "original_shape": (0, 0),
            "processed_shape": (0, 0),
            "columns": [],
            "dtypes": {},
            "dataframe": [],
            "markdown": "",
            "text_representation": "Empty table",
            "chunks": [],
            "metadata": metadata or {}
        }

    def process_tables_from_file(
        self,
        file_path: Union[str, Path],
        table_format: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Process tables from a file.

        Args:
            file_path: Path to file containing tables
            table_format: Format of tables in file

        Returns:
            List of processed tables
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if table_format == "csv" or file_path.suffix.lower() == ".csv":
                # Simple CSV parsing
                with open(file_path, 'r', encoding='utf-8') as f:
                    import csv
                    reader = csv.reader(f)
                    table_data = list(reader)
                result = self.process_table(table_data, table_id=file_path.stem)
                return [result]

            elif table_format == "json" or file_path.suffix.lower() == ".json":
                data = FileUtils.load_json(file_path)

                if isinstance(data, list):
                    # List of tables
                    results = []
                    for i, table_data in enumerate(data):
                        result = self.process_table(table_data, table_id=f"{file_path.stem}_{i}")
                        results.append(result)
                    return results
                else:
                    # Single table
                    result = self.process_table(data, table_id=file_path.stem)
                    return [result]

            else:
                logger.warning(f"Unsupported table format: {table_format}. Excel files require pandas.")
                return []

        except Exception as e:
            logger.error(f"Error processing tables from file {file_path}: {e}")
            return []

    def save_processed_tables(
        self,
        tables: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ):
        """
        Save processed tables to file.

        Args:
            tables: List of processed tables
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        FileUtils.save_json({"tables": tables}, output_path)
        logger.info(f"Saved {len(tables)} processed tables to {output_path}")


# Convenience functions
def process_table_simple(table_data: Union[List[List[Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple table processing function.

    Args:
        table_data: Table data

    Returns:
        Processed table
    """
    processor = TableProcessor()
    return processor.process_table(table_data)


def convert_table_to_markdown(table_data: Union[List[List[Any]], Dict[str, Any]]) -> str:
    """
    Convert table to markdown format.

    Args:
        table_data: Table data

    Returns:
        Markdown table string
    """
    processor = TableProcessor()
    result = processor.process_table(table_data)
    return result.get("markdown", "")
