from flask import Blueprint, request, render_template, redirect
from .auth import login_required

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('base.html')

@bp.route('/location', methods=['GET','POST'])
@login_required
def location():
    if request.method == 'POST':
        selected_choice = request.form['choice']
        # You can do whatever you want with the selected choice here
        print(f"You selected: {selected_choice}")
        return render_template('/main')
    return render_template('landing/location.html')

@bp.route('/landing')
@login_required
def landing():
    return render_template('landing/landing.html')