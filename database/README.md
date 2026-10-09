# Database

The application initializes `schema.sql` automatically on startup. The default persistent SQLite database is `instance/neurovista.sqlite3`; set `DATABASE_PATH` to change it. Records store upload metadata, SHA-256, status, and—only after an actual model run—the model output. Image bytes are stored in `instance/uploads/` as normalized PNG files.

Back up the database and upload directory together. Delete a scan from the UI/API to remove both its row and image. For disposal, stop the app and remove the entire configured database and upload directories. There is no cloud sync or automatic retention policy.
