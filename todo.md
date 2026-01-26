Add SPARQL support
If an entry fails it must not destroy all the workflow
Make sure the file lock is minimal (only lock when performing an IO task). handle the case of dead lock files (left over from previous time)
