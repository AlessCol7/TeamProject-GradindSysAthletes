import gradio as gr
import pyodbc
import datetime 


# Connect to Azure SQL Database
def connect_to_db():
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=tcp:atish.database.windows.net,1433;'
                      'Database=atish-LoginData;'
                      'Uid=atish;Pwd=13sql17_ctai;'
                      'Encrypt=yes;TrustServerCertificate=no;'
                      'Connection Timeout=30;')

    return conn
# Retrieve students data based on filters
def filter_students(email, start_date, end_date, min_score, max_score):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Prepare SQL query to fetch filtered students
    query = """
    SELECT email, date, model_score, teacher_score, teacher_feedback
    FROM students
    WHERE (email = ? OR ? = '')
    AND (date >= ? OR ? = '')
    AND (date <= ? OR ? = '')
    AND (model_score BETWEEN ? AND ?)
    """
    
    # Execute query with the filter parameters
    cursor.execute(query, (email, email, start_date, start_date, end_date, end_date, min_score, max_score))
    
    # Fetch and return results
    students = cursor.fetchall()
    conn.close()

    return [(student[0], student[1], student[2], student[3], student[4]) for student in students]

# Submit teacher's feedback and score
def submit_assessment(email, teacher_score, feedback):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Update the database with the teacher's feedback and score
    update_query = """
    UPDATE students
    SET teacher_score = ?, teacher_feedback = ?
    WHERE email = ?
    """
    
    cursor.execute(update_query, (teacher_score, feedback, email))
    conn.commit()
    conn.close()

    return f"Assessment submitted for {email} with score {teacher_score}."

# Gradio Components
email_dropdown = gr.Dropdown(choices=[], label="Select Student Email")
start_date_input =gr.Textbox(label="Start Date (YYYY-MM-DD)")
end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)")  
min_score_input = gr.Number(label="Min Model Score", value=0)
max_score_input = gr.Number(label="Max Model Score", value=5)

results_output = gr.DataFrame(headers=["Email", "Date", "Model Score", "Teacher Score", "Teacher Feedback"])

teacher_score_input = gr.Number(label="Teacher Score", minimum=1, maximum=5) 
teacher_feedback_input = gr.Textbox(label="Teacher Feedback", lines=2)

submit_button = gr.Button("Submit Assessment")

# Function to update dropdown dynamically
def update_email_dropdown():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT email FROM students")
    emails = [row[0] for row in cursor.fetchall()]
    conn.close()
    return emails

# Function to update results based on filters
def update_results(email, start_date, end_date, min_score, max_score):
    students = filter_students(email, start_date, end_date, min_score, max_score)
    return students

# Handle teacher's submission of feedback
def handle_submission(email, teacher_score, feedback):
    return submit_assessment(email, teacher_score, feedback)

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        gr.Column([email_dropdown, start_date_input, end_date_input, min_score_input, max_score_input])
    with gr.Row():
        results_output
    with gr.Row():
        teacher_score_input
        teacher_feedback_input
        submit_button

    # Setup actions for components
    email_dropdown.change(update_email_dropdown, outputs= email_dropdown)  # Update the email dropdown dynamically
    email_dropdown.change(update_results, inputs=[email_dropdown, start_date_input, end_date_input, min_score_input, max_score_input], outputs=results_output)
    submit_button.click(handle_submission, inputs=[email_dropdown, teacher_score_input, teacher_feedback_input], outputs=gr.Textbox())

# Launch Gradio app
demo.launch()


    # with gr.Tab("View Results"):
    #     results_output = gr.HTML(label="Video Results")
    #     student_email_input = gr.Dropdown(label="Select Student Email *", choices=[])
    #     refresh_dropdown_btn = gr.Button("Refresh Emails")
    #     get_results_btn = gr.Button("Get Results")

    #     # Function to fetch emails and recreate the dropdown
    #     def update_student_dropdown():
    #         # Connect to the database and fetch emails
    #         conn = connect_to_db()
    #         cursor = conn.cursor()

    #         cursor.execute("SELECT Role FROM Users WHERE Email = ?", (current_user_email,))
    #         user_role = cursor.fetchone()

    #         if user_role and user_role[0] == "teacher":
    #             # Teacher: Show all student emails
    #             cursor.execute("SELECT Email FROM Users WHERE Role = 'student'")
    #             student_emails = cursor.fetchall()
    #             student_email_input.choices = [email[0] for email in student_emails]
    #         elif user_role and user_role[0] == "student":
    #             # Student: Only show their own email
    #             student_email_input.choices = [current_user_email]
            
    #         # No need for a refresh button, just set the choices for the dropdown
    #         # Ensure the dropdown is reset (optional)
    #         student_email_input.value = None


    #     # Function to display results based on selected email
    #     def display_results(selected_email):
    #         if current_user_email is None:
    #             return "Error: Please log in to view results."

    #         conn = connect_to_db()
    #         cursor = conn.cursor()

    #         cursor.execute("SELECT Role FROM Users WHERE Email = ?", (current_user_email,))
    #         user_role = cursor.fetchone()

    #         if user_role:
    #             # Fetch the UserID for the selected email
    #             cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (selected_email,))
    #             user_id_result = cursor.fetchone()

    #             if user_id_result:
    #                 user_id = user_id_result[0]
                    
    #                 # Teacher can view any student's videos
    #                 if user_role[0] == "teacher":
    #                     cursor.execute("""
    #                         SELECT FileName, SportBranch, GuessGrade, UploadTime 
    #                         FROM Videos 
    #                         WHERE UserID = ? 
    #                         ORDER BY UploadTime DESC
    #                     """, (user_id,))
    #                     uploaded_videos = cursor.fetchall()

    #                     if uploaded_videos:
    #                         results_html = "<h3>Uploaded Videos</h3>"
    #                         for video in uploaded_videos:
    #                             results_html += f"""
    #                             <p><strong>Sport:</strong> {video[1]}</p>
    #                             <p><strong>Guess Grade:</strong> {video[2]}</p>
    #                             <p><strong>Uploaded On:</strong> {video[3]}</p>
    #                             <p><strong>File Name:</strong> {video[0]}</p>
    #                             <hr>
    #                             """
    #                         return results_html
    #                     else:
    #                         return "No videos uploaded by the selected student."

    #                 # Student can only view their own videos
    #                 elif user_role[0] == "student" and selected_email == current_user_email:
    #                     cursor.execute("""
    #                         SELECT FileName, SportBranch, GuessGrade, UploadTime 
    #                         FROM Videos 
    #                         WHERE UserID = ? 
    #                         ORDER BY UploadTime DESC
    #                     """, (user_id,))
    #                     uploaded_videos = cursor.fetchall()

    #                     if uploaded_videos:
    #                         results_html = "<h3>Your Uploaded Videos</h3>"
    #                         for video in uploaded_videos:
    #                             results_html += f"""
    #                             <p><strong>Sport:</strong> {video[1]}</p>
    #                             <p><strong>Guess Grade:</strong> {video[2]}</p>
    #                             <p><strong>Uploaded On:</strong> {video[3]}</p>
    #                             <p><strong>File Name:</strong> {video[0]}</p>
    #                             <hr>
    #                             """
    #                         return results_html
    #                     else:
    #                         return "You have not uploaded any videos."
                
    #             else:
    #                 return "Error: Student not found."
    #         else:
    #             return "Error: Unable to determine your role."


        # Define your Gradio UI components
        # student_email_input = gr.Dropdown(label="Select Student Email", choices=[])

        # # Call the update function initially to populate the dropdown
        # update_student_dropdown()

        # # Display results based on the selected email
        # student_email_input.change(display_results, inputs=student_email_input, outputs="results_html")



        # # Link refresh button to recreate the dropdown
        # refresh_dropdown_btn.click(
        #     update_student_dropdown,
        #     inputs=[],
        #     outputs=student_email_input
        # )

        # # Link results button to fetch and display results
        # get_results_btn.click(
        #     display_results,
        #     inputs=[student_email_input],
        #     outputs=results_output
        # )