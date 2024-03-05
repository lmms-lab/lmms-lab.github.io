/*===== MENU SHOW =====*/ 
const showMenu = (toggleId, navId) =>{
    const toggle = document.getElementById(toggleId),
    nav = document.getElementById(navId)

    if(toggle && nav){
        toggle.addEventListener('click', ()=>{
            nav.classList.toggle('show')
        })
    }
}
showMenu('nav-toggle','nav-menu')

/*==================== REMOVE MENU MOBILE ====================*/
const navLink = document.querySelectorAll('.nav__link')

function linkAction(){
    const navMenu = document.getElementById('nav-menu')
    // When we click on each nav__link, we remove the show-menu class
    navMenu.classList.remove('show')
}
navLink.forEach(n => n.addEventListener('click', linkAction))

/*==================== SCROLL SECTIONS ACTIVE LINK ====================*/
const sections = document.querySelectorAll('section[id]')

const scrollActive = () =>{
    const scrollDown = window.scrollY

  sections.forEach(current =>{
        const sectionHeight = current.offsetHeight,
              sectionTop = current.offsetTop - 58,
              sectionId = current.getAttribute('id'),
              sectionsClass = document.querySelector('.nav__menu a[href*=' + sectionId + ']')
        
        if(scrollDown > sectionTop && scrollDown <= sectionTop + sectionHeight){
            sectionsClass.classList.add('active-link')
        }else{
            sectionsClass.classList.remove('active-link')
        }
    })
}
window.addEventListener('scroll', scrollActive)

/*===== SCROLL REVEAL ANIMATION =====*/
const sr = ScrollReveal({
    origin: 'top',
    distance: '60px',
    duration: 2000,
    delay: 200,
//     reset: true
});

sr.reveal('.home__data, .about__img, .skills__subtitle, .skills__text',{}); 
sr.reveal('.home__img, .about__subtitle, .about__text, .skills__img',{delay: 400}); 
sr.reveal('.home__social-icon',{ interval: 200}); 
sr.reveal('.skills__data, .work__img, .contact__input',{interval: 200}); 
sr.reveal('.author__data, .author__img, .author__name, .author__description',{interval: 100});
sr.reveal('.lmms-eval__content, .advantage-description',{interval: 200});

document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('.advantage-btn');
    const descriptionElement = document.getElementById('advantage-description');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            buttons.forEach(btn => btn.classList.remove('advantage-btn--active')); // Remove active class from all buttons
            this.classList.add('advantage-btn--active'); // Add active class to the clicked button

            const advantage = this.getAttribute('data-advantage');
            switch (advantage) {
                case 'fast':
                    descriptionElement.textContent = 'Detailed description of how it is fast.';
                    break;
                case 'consistent':
                    descriptionElement.textContent = 'Explanation of its consistency.';
                    break;
                // Add more cases as needed
                default:
                    descriptionElement.textContent = '';
            }
        });
    });
    // Set initial state for "Fast"
    document.querySelector('.advantage-btn[data-advantage="fast"]').classList.add('advantage-btn--active');
    descriptionElement.textContent = 'Detailed description of how it is fast.';
});