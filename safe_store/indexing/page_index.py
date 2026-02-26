from typing import List, Dict, Any, Optional, Union
import sqlite3
import json
from ..core.exceptions import PageNotFoundError, PageIndexError

class PageIndex:
    """Handles hierarchical page structures within a SafeStore database."""
    
    def __init__(self, store):
        self.store = store

    @property
    def conn(self) -> sqlite3.Connection:
        return self.store.conn

    def add_page(self, doc_id: int, title: str, content: Optional[str] = None, 
                 parent_id: Optional[int] = None, page_order: int = 0, 
                 metadata: Optional[Dict[str, Any]] = None) -> int:
        """Adds a page to the hierarchy."""
        level = 0
        if parent_id:
            parent = self.get_page(parent_id)
            level = parent['level'] + 1

        meta_json = json.dumps(metadata) if metadata else None
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO pages (doc_id, parent_id, title, content, level, page_order, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, parent_id, title, content, level, page_order, meta_json))
        return cursor.lastrowid

    def get_page(self, page_id: int) -> Dict[str, Any]:
        """Fetches a single page and its immediate children info."""
        cursor = self.conn.execute("""
            SELECT page_id, doc_id, parent_id, title, content, level, page_order, metadata 
            FROM pages WHERE page_id = ?
        """, (page_id,))
        row = cursor.fetchone()
        if not row:
            raise PageNotFoundError(f"Page {page_id} not found.")
        
        return self._row_to_dict(row)

    def get_children(self, page_id: int) -> List[Dict[str, Any]]:
        """Returns immediate children of a page."""
        cursor = self.conn.execute("""
            SELECT page_id, doc_id, parent_id, title, content, level, page_order, metadata 
            FROM pages WHERE parent_id = ? ORDER BY page_order ASC
        """, (page_id,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_breadcrumbs(self, page_id: int) -> List[Dict[str, Any]]:
        """Returns the path from root to the current page."""
        path = []
        current_id = page_id
        while current_id:
            cursor = self.conn.execute("SELECT page_id, parent_id, title FROM pages WHERE page_id = ?", (current_id,))
            row = cursor.fetchone()
            if not row: break
            path.insert(0, {"page_id": row[0], "title": row[2]})
            current_id = row[1]
        return path

    def get_tree(self, doc_id: int, parent_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Recursively builds the full tree structure for a document or sub-tree."""
        query = "SELECT page_id, doc_id, parent_id, title, content, level, page_order, metadata FROM pages WHERE doc_id = ? AND "
        params = [doc_id]
        
        if parent_id is None:
            query += "parent_id IS NULL"
        else:
            query += "parent_id = ?"
            params.append(parent_id)
        
        query += " ORDER BY page_order ASC"
        
        cursor = self.conn.execute(query, tuple(params))
        nodes = []
        for row in cursor.fetchall():
            node = self._row_to_dict(row)
            node['children'] = self.get_tree(doc_id, node['page_id'])
            nodes.append(node)
        return nodes

    def _row_to_dict(self, row) -> Dict[str, Any]:
        return {
            "page_id": row[0],
            "doc_id": row[1],
            "parent_id": row[2],
            "title": row[3],
            "content": row[4],
            "level": row[5],
            "page_order": row[6],
            "metadata": json.loads(row[7]) if row[7] else {}
        }

    def import_markdown_as_tree(self, doc_id: int, markdown_text: str):
        """
        Parses markdown headers (#, ##, ###) and creates a hierarchical tree.
        """
        import re
        lines = markdown_text.split('\n')
        stack = [(0, None)] # (level, parent_id)
        
        current_content = []
        current_title = "Introduction"
        
        def flush_page(title, content, parent_id, order):
            return self.add_page(doc_id, title, "\n".join(content), parent_id, order)

        order = 0
        for line in lines:
            header_match = re.match(r'^(#+)\s+(.*)', line)
            if header_match:
                # Flush previous page
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                # Pop stack until we find the parent (one level up)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                
                parent_id = stack[-1][1] if stack else None
                new_id = self.add_page(doc_id, title, "", parent_id, order)
                stack.append((level, new_id))
                order += 1
            else:
                # Add text to the current leaf in the stack
                if stack[-1][1]:
                    # This is inefficient for huge files, but safe for metadata/TOC
                    self.conn.execute(
                        "UPDATE pages SET content = IFNULL(content, '') || ? WHERE page_id = ?",
                        (line + "\n", stack[-1][1])
                    )
        self.conn.commit()

    def import_csv_as_flat_list(self, doc_id: int, csv_content: str, title_column: str):
        """
        Imports CSV rows as child pages under the document root.
        """
        import csv
        import io
        reader = csv.DictReader(io.StringIO(csv_content))
        for i, row in enumerate(reader):
            title = row.get(title_column, f"Row {i}")
            content = json.dumps(row, indent=2)
            self.add_page(doc_id, title, content, parent_id=None, page_order=i, metadata={"type": "csv_row"})
        self.conn.commit()