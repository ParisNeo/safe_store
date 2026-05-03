# API Documentation

## Authentication
All API requests require a valid authentication token.

## Endpoints

### GET /documents
Retrieve a list of all documents in the store.

### POST /documents
Add a new document to the store with metadata.

### DELETE /documents/{id}
Remove a document by its unique identifier.

## Error Handling

### 400 Bad Request
The request was malformed or missing required parameters.

### 404 Not Found
The requested resource does not exist.

### 500 Internal Server Error
An unexpected error occurred on the server.

## Rate Limiting
API calls are limited to 100 requests per minute per API key.
