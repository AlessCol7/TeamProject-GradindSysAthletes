import gradio as gr
import pyodbc
import hashlib
import re


def connect_to_db():
    # Connect to the database
    conn = pyodbc.connect(
        'Driver={ODBC Driver 17 for SQL Server};'
        'Server=tcp:atish.database.windows.net,1433;'
        'Database=atish-LoginData;'
        'Uid=atish;Pwd=13sql17_ctai;'
        'Encrypt=yes;TrustServerCertificate=no;'
        'Connection Timeout=30;'
    )
    return conn


def get_rubrics_from_db(sport_branch):
    # Fetch rubrics for the selected sport branch
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT Criteria, Score FROM Rubrics WHERE SportBranch = ?", (sport_branch,))
    rows = cursor.fetchall()
    conn.close()

    # Convert to an HTML table
    if rows:
        table = "<table><tr><th>Criteria</th><th>Score</th></tr>"
        for criteria, score in rows:
            table += f"<tr><td>{criteria}</td><td>{score}</td></tr>"
        table += "</table>"
        return table
    else:
        return "No rubric available for this sport branch."


def update_rubrics_in_db(sport_branch, new_rubrics):
    # Update rubrics in the database
    conn = connect_to_db()
    cursor = conn.cursor()

    # Delete existing rubrics for the sport branch
    cursor.execute("DELETE FROM Rubrics WHERE SportBranch = ?", (sport_branch,))

    # Insert new rubrics
    for criteria, score in new_rubrics:
        cursor.execute("INSERT INTO Rubrics (SportBranch, Criteria, Score) VALUES (?, ?, ?)",
                       (sport_branch, criteria, score))

    conn.commit()
    conn.close()
    return "Rubrics updated successfully!"


def role_based_view(email):
    # Determine the role (student or teacher)
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT Role FROM Users WHERE Email = ?", (email,))
    role = cursor.fetchone()
    conn.close()
    return role[0] if role else None


# Main Gradio app
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics Evaluation System 🏃‍♂️")

    with gr.Tab("Login"):
        email_input_log = gr.Textbox(label="Email *")
        password_input_log = gr.Textbox(label="Password *", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)

        def validate_login(email, password):
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("SELECT Password, Role FROM Users WHERE Email = ?", (email,))
            user_data = cursor.fetchone()

            if user_data:
                stored_password, role = user_data
                input_password_hashed = hashlib.sha256(password.encode()).hexdigest()
                if stored_password == input_password_hashed:
                    return f"Welcome, {role}!", role
            return "Invalid email or password", None

        login_btn.click(
            validate_login,
            inputs=[email_input_log, password_input_log],
            outputs=[login_output, gr.State()]
        )

    with gr.Tab("Rubrics"):
        role_state = gr.State()  # Store the user's role (teacher or student)

        sport_branch_input = gr.Dropdown(
            ["", "Sprint Start", "Sprint Running", "Shot Put", "Relay Receiver", "Long Jump", "Javelin", "High Jump", "Discus Throw", "Hurdling"],
            label="Select Sport Branch"
        )
        rubrics_output = gr.HTML(label="Rubrics")
        view_rubrics_btn = gr.Button("View Rubrics")

        def view_rubrics(sport_branch, role):
            if role == "student":
                return get_rubrics_from_db(sport_branch)
            elif role == "teacher":
                return get_rubrics_from_db(sport_branch)
            return "Error: Unauthorized access."

        # Fix: Ensure correct inputs (Dropdown for sport branch, State for role)
        view_rubrics_btn.click(
            view_rubrics,
            inputs=[sport_branch_input, role_state],  # Correctly mapping the dropdown and role
            outputs=rubrics_output
        )

        with gr.Row(visible=False) as edit_rubrics_row:
            edit_rubrics_input = gr.Dataframe(headers=["Criteria", "Score"], label="Edit Rubrics")
            save_rubrics_btn = gr.Button("Save Rubrics")
            edit_output = gr.Textbox(label="Edit Status")

            save_rubrics_btn.click(
                lambda sport_branch, rubrics: update_rubrics_in_db(sport_branch, rubrics),
                inputs=[sport_branch_input, edit_rubrics_input],  # Ensure correct input mapping
                outputs=edit_output
            )

        def show_edit_rubrics(role):
            # Only teachers can edit rubrics
            return role == "teacher"

        role_state.change(
            show_edit_rubrics,
            inputs=[role_state],
            outputs=[edit_rubrics_row]
        )


if __name__ == "__main__":
    athletics_app.launch(debug=True)
